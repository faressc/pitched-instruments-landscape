import utils.debug

# This is needed to avoid errors from cuda when using multiprocessing
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from pathlib import Path
import os
from typing import Sequence, Iterable, Tuple, List, ByteString
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

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
    path = Path(path_str)
    audios = []
    if not path.is_dir():
        raise ValueError(f"Path {path} is not a directory")
    for ext in extensions:
        audios.append(path.rglob(f'*.{ext}'))
        audios.append(path.rglob(f'*.{ext.upper()}'))
    # rglob returns a generator list of lists, so we need to flatten it
    for audio in audios:
        for a in audio:
            yield a

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
    return model

def infer_encodec(audio: np.ndarray, model: EncodecModel) -> Tuple[torch.Tensor, torch.Tensor]:
    audio = audio.astype(np.float32) / (2**15 - 1)
    device = next(model.parameters()).device
    x = torch.from_numpy(audio).to(device)
    x = x.view(1, 1, -1)
    encoder_outputs = model.encode(x, bandwidth=24.0)
    audio_codes = encoder_outputs.audio_codes.view(1, encoder_outputs.audio_codes.size(2), encoder_outputs.audio_codes.size(3)) # type: ignore
    audio_codes = rearrange(audio_codes, 'b q t -> q b t')
    embeddings = model.quantizer.decode(audio_codes)
    return encoder_outputs.audio_codes.detach().cpu().numpy(), embeddings.detach().cpu().numpy() # type: ignore

def process_batch(batch_items: List[Tuple[int, str]], sample_rate: int, model_name: str, device: torch.device = torch.device("cpu")) -> List[Tuple[int, ByteString, float]]:
    """Process a batch of audio files in a separate process"""

    # Set the number of threads to 1 to avoid thread contention
    torch.set_num_threads(1)

    # Setup local model for this process
    model = load_model_with_warnings_filtered(model_name, device)
    
    # Process all audio files in the batch and collect results
    results = []
    for index, path in batch_items:
        audio_data = ffmpeg.load_audio_file(path, sample_rate=sample_rate)
        
        # Process audio but don't write to database
        meta_audio_file, duration = process_audio_data(path, audio_data, sample_rate, model)
        results.append((index, meta_audio_file, duration))
    
    return results

def process_audio_data(path: str, audio: Tuple[np.ndarray, float, int], sample_rate: int, model: EncodecModel) -> Tuple[ByteString, float]:
    """Process audio data without writing to database"""
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

    audio_codes, embeddings = infer_encodec(audio_data, model)

    audio_codes = MetaAudioFile.EncodecOutputs.AudioCodes(
        data=audio_codes.tobytes(), # type: ignore
        dtype=MetaAudioFile.DType.INT64,
        shape=audio_codes.shape
    )
    embeddings = MetaAudioFile.EncodecOutputs.Embeddings(
        data=embeddings.tobytes(), # type: ignore
        dtype=MetaAudioFile.DType.FLOAT32,
        shape=embeddings.shape
    )
    encodec_outputs = MetaAudioFile.EncodecOutputs(
        audio_codes=audio_codes,
        embeddings=embeddings
    )

    meta_audio_file = MetaAudioFile(
        audio_file=audio_file,
        metadata=metadata,
        encoder_outputs=encodec_outputs
    )

    return meta_audio_file.SerializeToString(), duration

def get_db_size(env):
    """Get current size of LMDB database in bytes"""
    with env.begin() as txn:
        stats = env.stat()
        return stats['psize'] * (stats['leaf_pages'] + stats['branch_pages'] + stats['overflow_pages'])

def check_and_resize_db(env, db_size, current_usage_ratio=0.8, force=False):
    """Check if database is nearing capacity and resize if needed"""
    current_size_bytes = get_db_size(env)
    max_size_bytes = db_size * 1024**3
    
    # If database has reached the threshold (80% by default), increase size
    if force or current_size_bytes > current_usage_ratio * max_size_bytes:
        new_size_gb = db_size * 2
        print(f"Resizing database from {db_size}GB to {new_size_gb}GB")
        db_size = new_size_gb
        env.set_mapsize(new_size_gb * 1024**3)
    return

def write_to_database(index: int, meta_audio_file: ByteString, env: lmdb.Environment, db_size) -> None:
    """Write data to LMDB database from main process"""
    index_str = f"{index:08}"
    # Maximum number of resize attempts
    max_resize_attempts = 3
    resize_attempts = 0

    while True:
        try:
            with env.begin(write=True) as txn:
                txn.put(index_str.encode(), meta_audio_file)
            break  # Success, exit the loop
        except lmdb.MapFullError:
            resize_attempts += 1
            if resize_attempts > max_resize_attempts:
                raise Exception(f"Failed to write to database after {max_resize_attempts} resize attempts")
            check_and_resize_db(env, db_size, force=True)
        except Exception as e:
            raise Exception(f"Error writing to database: {str(e)}")

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
        with ProcessPoolExecutor(max_workers=cfg.preprocess.num_workers) as executor:
            # Submit all batches for processing
            futures = []
            for batch in batches:
                future = executor.submit(
                    process_batch_partial, 
                    batch
                )
                futures.append(future)
            
            # Process results as they complete
            pbar = tqdm(total=len(audio_files))
            for future in as_completed(futures):
                try:
                    results = future.result()
                    # Write processed items to database from main process
                    for index, meta_audio_file, duration in results:
                        # Check and resize database if needed before each write
                        check_and_resize_db(env, db_size)
                        # Write to database from main process
                        write_to_database(index, meta_audio_file, env, db_size)
                        total_duration += duration
                        pbar.update(1)
                        hours, remainder = divmod(total_duration, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        pbar.set_description(f"Total duration: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
                except Exception as e:
                    print(f"An error occurred: {str(e)}")
                    print(f"Traceback: {e.__traceback__}")
        
        # Close the database
        env.close()
        
        print(f"Final database size for {output_path}: {db_size}GB")
        print(f"Total duration of audio files in database ({output_path}): {total_duration} seconds")
        
        # Final consolidation of database if needed
        print(f"Finished processing files for {output_path}")

if __name__ == "__main__":
    main()
