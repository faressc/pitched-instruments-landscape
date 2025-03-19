import utils.debug

# This is needed to avoid errors from cuda when using multiprocessing
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from pathlib import Path
import os
from typing import Sequence, Iterable, Tuple, List
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

def search_for_audios(path: str, extensions: Sequence[str]) -> Iterable[Path]:
    path = Path(path)
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

def get_metadata(path: str) -> dict:
    path = Path(path)
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
    audio_codes = encoder_outputs.audio_codes.view(1, encoder_outputs.audio_codes.size(2), encoder_outputs.audio_codes.size(3))
    audio_codes = rearrange(audio_codes, 'b q t -> q b t')
    embeddings = model.quantizer.decode(audio_codes)
    return encoder_outputs.audio_codes.detach().cpu().numpy(), embeddings.detach().cpu().numpy()

def process_batch(batch_items, output_path, db_size, sample_rate, model_name, device):
    """Process a batch of audio files in a separate process"""

    # Set the number of threads to 1 to avoid thread contention
    # TODO: Find a better way to do this, maybe using torch.multiprocessing.set_start_method('spawn')?
    # because torch still spawns many threads even with torch.set_num_threads(1) allready when importing torch
    torch.set_num_threads(1)

    # Setup local model for this process
    model = load_model_with_warnings_filtered(model_name, device)
    
    # Open a local LMDB environment
    env = lmdb.open(
        output_path,
        map_size=db_size * 1024**3,
        readonly=False,
        lock=False  # Important for concurrent access
    )
    
    results = []
    for index, path in batch_items:
        audio_data = ffmpeg.load_audio_file(path, sample_rate=sample_rate)
        duration = preprocess_audio_file((index, path, audio_data), env, sample_rate, model)
        results.append((index, duration))
    
    env.close()
    return results

def preprocess_audio_file(audio: Tuple[int, Path, Tuple[Sequence[np.ndarray], float, int]], env: lmdb.Environment, sample_rate: int, model: EncodecModel) -> int:
    index, path, audio = audio
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
        data=audio_codes.tobytes(),
        dtype=MetaAudioFile.DType.INT64,
        shape=audio_codes.shape
    )

    embeddings = MetaAudioFile.EncodecOutputs.Embeddings(
        data=embeddings.tobytes(),
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

    index_str = f"{index:08}"
    with env.begin(write=True) as txn:
        txn.put(index_str.encode(), meta_audio_file.SerializeToString())

    return duration

def main():
    print("##### Starting Parallel Preprocessing Stage #####")

    # Load the parameters from the dictionary into variables
    cfg = OmegaConf.load("params.yaml")
    
    print(f"Using {cfg.preprocess.num_workers} worker processes with batch size {cfg.preprocess.batch_size}")

    input_paths = [cfg.preprocess.input_path_train, cfg.preprocess.input_path_valid, cfg.preprocess.input_path_test]
    output_paths = [cfg.preprocess.output_path_train, cfg.preprocess.output_path_valid, cfg.preprocess.output_path_test]
    database_sizes = [cfg.preprocess.max_db_size_train, cfg.preprocess.max_db_size_valid, cfg.preprocess.max_db_size_test]
    
    # Determine device configuration
    device = config.prepare_device(cfg.device)

    for input_path, output_path, database_size in zip(input_paths, output_paths, database_sizes):
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

        # Create a main LMDB database with the correct map size
        env = lmdb.open(
            output_path,
            map_size=database_size * 1024**3,
            # This is needed otherwise python crashes on the hpc
            # writemap=True
        )
        env.close()  # Close it immediately, we'll use it in the workers

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

        process_batch_partial = partial(process_batch, output_path=output_path, db_size=database_size, sample_rate=cfg.preprocess.sample_rate, model_name=cfg.preprocess.model_name, device=device)
        
        print(f"Split files into {len(batches)} batches for parallel processing")
        
        # Process batches in parallel
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
                    batch_results = future.result()
                    for _, duration in batch_results:
                        total_duration += duration
                        pbar.update(1)
                        hours, remainder = divmod(total_duration, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        pbar.set_description(f"Total duration: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
                except Exception as e:
                    print(f"An error occurred: {str(e)}")
        
        print(f"Total duration of audio files in database ({output_path}): {total_duration} seconds")
        
        # Final consolidation of database if needed
        print(f"Finished processing files for {output_path}")

if __name__ == "__main__":
    main()
