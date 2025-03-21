import utils.debug

# This is needed to avoid errors from cuda when using multiprocessing
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from pathlib import Path
import os
from typing import Sequence, Iterable, Tuple, List, ByteString
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
import warnings
import gc

from omegaconf import OmegaConf
import lmdb
import numpy as np
from transformers import EncodecModel, AutoProcessor
from einops import rearrange
from tqdm import tqdm
import json
import torch

import utils.ffmpeg_helper as ffmpeg
import utils.config as config
try:
    from proto.meta_audio_file_pb2 import MetaAudioFile
except:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from proto.meta_audio_file_pb2 import MetaAudioFile

def search_for_audios(path_str: str, extensions: Sequence[str]) -> Iterable[Path]:
    """
    Search for audio files with specified extensions in a directory.
    Uses generators to minimize memory usage.
    """
    path = Path(path_str)
    if not path.is_dir():
        raise ValueError(f"Path {path} is not a directory")
        
    # Use a generator expression to avoid building a list in memory
    for ext in extensions:
        # Search for lowercase extension
        for audio_path in path.rglob(f'*.{ext}'):
            yield audio_path
            
        # Search for uppercase extension
        for audio_path in path.rglob(f'*.{ext.upper()}'):
            yield audio_path

def get_metadata(path_str: str) -> dict:
    path = Path(path_str)
    filename = path.stem
    folder_path = path.parts[:-2]
    folder_path = Path(*folder_path)
    metadata_path = folder_path / "examples.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found for {path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata[filename]

def load_model_with_warnings_filtered(model_name, device):
    """
    Load the EncodecModel with warnings about tensor construction filtered out.
    """
    # Filter out specific PyTorch tensor construction warnings from transformers
    with warnings.catch_warnings():
        # Suppress warnings in this process
        warnings.filterwarnings(
            "ignore", 
            category=UserWarning,
            message=".*To copy construct from a tensor.*"
        )
        model = EncodecModel.from_pretrained(model_name)
        model.to(device)
        # Set model to eval mode to disable gradient tracking
        model.eval()
    return model

def infer_encodec(audio: np.ndarray, model: EncodecModel) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process audio through the EncodecModel with careful memory management.
    Returns numpy arrays of audio codes and embeddings.
    """
    # Convert audio to float32 and normalize
    audio = audio.astype(np.float32) / (2**15 - 1)
    device = next(model.parameters()).device
    
    try:
        # Create tensor and move to device
        x = torch.from_numpy(audio).to(device)
        x = x.view(1, 1, -1)
        
        # Free the numpy array memory
        del audio
        
        with torch.no_grad():  # Use no_grad to reduce memory usage
            # Encode in steps to manage memory better
            encoder_outputs = model.encode(x, bandwidth=24.0)
            
            # Release input tensor
            del x
            
            # Process audio codes
            audio_codes = encoder_outputs.audio_codes.view(
                1, encoder_outputs.audio_codes.size(2), encoder_outputs.audio_codes.size(3)
            ) # type: ignore
            audio_codes = rearrange(audio_codes, 'b q t -> q b t')
            
            # First extract and copy audio codes before generating embeddings
            audio_codes_np = encoder_outputs.audio_codes.detach().cpu().numpy().copy() # type: ignore
            
            # Release encoder outputs after extracting codes
            del encoder_outputs
            
            # Generate embeddings separately
            embeddings = model.quantizer.decode(audio_codes)
            embeddings_np = embeddings.detach().cpu().numpy().copy() # type: ignore
            
            # Release tensors immediately
            del embeddings, audio_codes
        
        # Ensure arrays are contiguous to prevent memory leaks
        audio_codes_np = np.ascontiguousarray(audio_codes_np)
        embeddings_np = np.ascontiguousarray(embeddings_np)
        
        return audio_codes_np, embeddings_np
        
    finally:
        # Clear CUDA cache if using GPU
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()

def process_batch(batch_items: List[Tuple[int, str]], sample_rate: int, model_name: str, device: torch.device = torch.device("cpu")) -> List[Tuple[int, ByteString, float]]:
    """Process a batch of audio files in a separate process"""

    # Set the number of threads to 1 to avoid thread contention
    torch.set_num_threads(1)

    # Setup local model for this process
    model = load_model_with_warnings_filtered(model_name, device)
    
    # Process all audio files in the batch and collect results
    results = []
    try:
        for index, path in batch_items:
            audio_data = ffmpeg.load_audio_file(path, sample_rate=sample_rate)
            
            # Process audio but don't write to database
            meta_audio_file, duration = process_audio_data(path, audio_data, sample_rate, model)
            results.append((index, meta_audio_file, duration))
            
    finally:
        # Clean up model resources
        del model
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results

def process_audio_data(path: str, audio: Tuple[np.ndarray, float, int], sample_rate: int, model: EncodecModel) -> Tuple[ByteString, float]:
    """Process audio data without writing to database"""
    try:
        audio_data, duration, num_channels = audio
        audio_file = MetaAudioFile.AudioFile(
            data=audio_data.tobytes(),
            sample_rate=sample_rate,
            dtype=MetaAudioFile.DType.INT16,
            num_channels=num_channels,
            num_samples=audio_data.shape[0]//num_channels
        )

        metadata_dict = get_metadata(path)
        metadata = MetaAudioFile.Metadata(
            note=metadata_dict["note"],
            note_str=metadata_dict["note_str"],
            instrument=metadata_dict["instrument"],
            instrument_str=metadata_dict["instrument_str"],
            pitch=metadata_dict["pitch"],
            velocity=metadata_dict["velocity"],
            qualities=metadata_dict["qualities"],
            family=metadata_dict["instrument_family"],
            source=metadata_dict["instrument_source"],
        )

        # Process audio with encodec model
        audio_codes, embeddings = infer_encodec(audio_data, model)

        # Convert to protocol buffer format
        audio_codes_pb = MetaAudioFile.EncodecOutputs.AudioCodes(
            data=audio_codes.tobytes(), # type: ignore
            dtype=MetaAudioFile.DType.INT64,
            shape=audio_codes.shape
        )
        embeddings_pb = MetaAudioFile.EncodecOutputs.Embeddings(
            data=embeddings.tobytes(), # type: ignore
            dtype=MetaAudioFile.DType.FLOAT32,
            shape=embeddings.shape
        )
        encodec_outputs = MetaAudioFile.EncodecOutputs(
            audio_codes=audio_codes_pb,
            embeddings=embeddings_pb
        )

        # Create final protocol buffer
        meta_audio_file = MetaAudioFile(
            audio_file=audio_file,
            metadata=metadata,
            encoder_outputs=encodec_outputs
        )

        # Serialize and return
        serialized_data = meta_audio_file.SerializeToString()
        
        # Clean up large objects to free memory
        del audio_data, audio_codes, embeddings
        del audio_file, metadata_dict, metadata, audio_codes_pb, embeddings_pb, encodec_outputs, meta_audio_file
        
        return serialized_data, duration
    except Exception as e:
        # Clean up in case of error
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise e

def get_db_size(env):
    """Get current size of LMDB database in bytes"""
    with env.begin() as txn:
        stats = env.stat()
        return stats['psize'] * (stats['leaf_pages'] + stats['branch_pages'] + stats['overflow_pages'])

def check_and_resize_db(env, db_size, current_usage_ratio=0.8, force=False) -> int:
    """Check if database is nearing capacity and resize if needed"""
    current_size_bytes = get_db_size(env)
    max_size_bytes = db_size * 1024**3
    
    # If database has reached the threshold (80% by default), increase size
    if force or current_size_bytes > current_usage_ratio * max_size_bytes:
        new_size_gb = db_size * 2
        print(f"Resizing database from {db_size}GB to {new_size_gb}GB")
        db_size = new_size_gb
        env.set_mapsize(new_size_gb * 1024**3)
    return db_size

def write_to_database(index: int, meta_audio_file: ByteString, env: lmdb.Environment, db_size) -> int:
    """Write data to LMDB database from main process"""
    index_str = f"{index:08}"
    # Maximum number of resize attempts
    max_resize_attempts = 3
    resize_attempts = 0

    while True:
        try:
            # Ensure transaction is properly committed/aborted
            txn = env.begin(write=True)
            try:
                txn.put(index_str.encode(), meta_audio_file)
                txn.commit()
                break  # Success, exit the loop
            except Exception as e:
                # Make sure to abort the transaction on error
                txn.abort()
                if isinstance(e, lmdb.MapFullError):
                    raise e
                else:
                    raise Exception(f"Error writing to database: {str(e)}")
        except lmdb.MapFullError:
            resize_attempts += 1
            if resize_attempts > max_resize_attempts:
                raise Exception(f"Failed to write to database after {max_resize_attempts} resize attempts")
            db_size = check_and_resize_db(env, db_size, force=True)
        except Exception as e:
            raise Exception(f"Error writing to database: {str(e)}")
    return db_size

def main():
    print("##### Starting Parallel Preprocessing Stage #####")

    # Load the parameters from the dictionary into variables
    cfg = OmegaConf.load("params.yaml")
    
    print(f"Using {cfg.preprocess.num_workers} worker processes with batch size {cfg.preprocess.batch_size}")

    input_paths = [cfg.preprocess.input_path_train, cfg.preprocess.input_path_valid, cfg.preprocess.input_path_test]
    output_paths = [cfg.preprocess.output_path_train, cfg.preprocess.output_path_valid, cfg.preprocess.output_path_test]
    
    # Determine device configuration
    device = config.prepare_device(cfg.device)

    for input_path, output_path in zip(input_paths, output_paths):
        print(f"Creating LMDB database at {output_path}")

        if not os.path.isdir(input_path):
            raise FileNotFoundError(f"Input path {input_path} does not exist")
        try:
            os.makedirs(output_path)
        except FileExistsError:
            print(f"Output path {output_path} already exists, skipping")
        
        # Create output directory structure
        output_dir = os.path.join(*os.path.split(output_path)[:-1])
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        print(f"Searching for audio files in {input_path}")
        # Search for audio files
        audio_files = search_for_audios(input_path, cfg.preprocess.ext)
        audio_files = list(map(lambda p: (str(p)), audio_files))
        audio_files = list(map(os.path.abspath, audio_files))

        print(f"Found {len(audio_files)} audio files")
        if len(audio_files) == 0:
            print(f"No audio files found in {input_path}")
            raise FileNotFoundError(f"No audio files found in {input_path}")

        # Create indexed pairs for processing
        indexed_files = list(enumerate(audio_files))
        
        # Split files into batches for parallel processing
        batches = [indexed_files[i:i + cfg.preprocess.batch_size] for i in range(0, len(indexed_files), cfg.preprocess.batch_size)]

        process_batch_partial = partial(process_batch, sample_rate=cfg.preprocess.sample_rate, model_name=cfg.preprocess.model_name, device=device)
        
        print(f"Split files into {len(batches)} batches for parallel processing")

        # Initial database size - only needed in main process
        db_size = cfg.preprocess.initial_db_size
        
        # Open LMDB environment in main process only
        env = lmdb.open(
            output_path,
            map_size=db_size * 1024**3,
            writemap=cfg.preprocess.db_writemap
        )
        
        total_duration = 0
        completed_count = 0
        batch_size = cfg.preprocess.batch_size  # For monitoring
        
        # Process files in smaller chunks to minimize memory consumption
        with ProcessPoolExecutor(max_workers=cfg.preprocess.num_workers) as executor:
            # Submit all batches for processing, but limit how many are in memory at once
            max_queued_batches = min(cfg.preprocess.num_workers * 2, len(batches))
            active_futures = set()
            batch_queue = list(batches)  # Create a queue of batches to process
            
            # Initial submission of batches
            for _ in range(min(max_queued_batches, len(batch_queue))):
                if batch_queue:
                    batch = batch_queue.pop(0)
                    future = executor.submit(process_batch_partial, batch)
                    active_futures.add(future)
            
            # Process results and submit new batches as they complete
            pbar = tqdm(total=len(audio_files))
            while active_futures:
                # Get the first completed future
                done, active_futures = wait(active_futures, 
                                           return_when=FIRST_COMPLETED)
                
                # Process completed futures
                for future in done:
                    try:
                        results = future.result()
                        
                        # Immediately submit a new batch if available
                        if batch_queue:
                            batch = batch_queue.pop(0)
                            new_future = executor.submit(process_batch_partial, batch)
                            active_futures.add(new_future)
                        
                        # Write processed items to database from main process
                        for index, meta_audio_file, duration in results:
                            # Check and resize database if needed before each write
                            db_size = check_and_resize_db(env, db_size)
                            # Write to database from main process
                            db_size = write_to_database(index, meta_audio_file, env, db_size)
                            total_duration += duration
                            pbar.update(1)
                            
                            # Update progress description with time
                            hours, remainder = divmod(total_duration, 3600)
                            minutes, seconds = divmod(remainder, 60)
                            completed_count += 1
                            pbar.set_description(
                                f"Processed: {completed_count}/{len(audio_files)} | "
                                f"Duration: {int(hours):02}:{int(minutes):02}:{int(seconds):02}"
                            )
                            
                            # Release memory for this item immediately
                            del meta_audio_file
                        
                        # Clear results list
                        del results
                        
                    except Exception as e:
                        print(f"An error occurred: {str(e)}")
                        print(f"Traceback: {e.__traceback__}")
                        
                        # Still submit new batch on error to keep workers busy
                        if batch_queue:
                            batch = batch_queue.pop(0)
                            new_future = executor.submit(process_batch_partial, batch)
                            active_futures.add(new_future)
                    
                    # Force garbage collection after each batch
                    gc.collect()
                    
                    # Clear any CUDA memory if available
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        # Close the database
        env.close()
        
        print(f"Final database size for {output_path}: {db_size}GB")
        print(f"Total duration of audio files in database ({output_path}): {total_duration} seconds")
        
        # Final consolidation of database if needed
        print(f"Finished processing files for {output_path}")

if __name__ == "__main__":
    main()
