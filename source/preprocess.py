import utils.debug

from pathlib import Path
import os
from typing import Sequence, Iterable, Tuple
from functools import partial

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
    print("##### Starting Preprocessing Stage #####")

    # Load the parameters from the dictionary into variables
    cfg = OmegaConf.load("params.yaml")

    print(f"Creating LMDB database at {cfg.preprocess.output_path_valid}")
    # The LMDB database will itself create the final directory, as it has multiple files
    # The * operator is used to unpack the tuple into single elements
    output_dir = os.path.join(*os.path.split(cfg.preprocess.output_path_valid)[:-1])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Create a new LMDB database
    env = lmdb.open(
        cfg.preprocess.output_path_valid,
        map_size=cfg.preprocess.max_db_size * 1024**3,
        # This is needed otherwise python crashes on the hpc
        writemap=True
    )

    print("Searching for audio files")
    # Search for audio files
    audio_files = search_for_audios(cfg.preprocess.input_path_valid, cfg.preprocess.ext)
    audio_files = map(str, audio_files)
    audio_files = map(os.path.abspath, audio_files)

    # Evaluate the generator
    audio_files = list(audio_files)
    print(f"Found {len(audio_files)} audio files")
    if len(audio_files) == 0:
        print(f"No audio files found in {cfg.preprocess.input_path_valid}")

    # Load and process the audio files
    partial_load_audio_file = partial(ffmpeg.load_audio_file, sample_rate=cfg.preprocess.sample_rate)
    audio_data = map(partial_load_audio_file, audio_files)
    audio = zip(range(len(audio_files)), audio_files, audio_data)

    # Load the model
    model = EncodecModel.from_pretrained(cfg.preprocess.model_name)
    device = config.prepare_device(cfg.device)
    model.to(device)
    
    processor = map(partial(preprocess_audio_file, env=env, sample_rate=cfg.preprocess.sample_rate, model=model), audio)

    pbar = tqdm(processor, total=len(audio_files))
    total_duration = 0
    for duration in pbar:
        total_duration += duration
        hours, remainder = divmod(total_duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        pbar.set_description(f"Total duration: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

    print(f"Total duration of audio files: {total_duration} seconds")

if __name__ == "__main__":
    main()