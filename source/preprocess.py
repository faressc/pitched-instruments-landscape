import utils.debug

from pathlib import Path
import os
import utils.ffmpeg_helper as ffmpeg

from omegaconf import OmegaConf
import lmdb
import numpy as np
from transformers import EncodecModel, AutoProcessor
from einops import rearrange
from typing import Sequence, Iterable

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

    for audio_file in audio_files:
        print(f"Processing {audio_file}")
        audio, duration, num_channels = ffmpeg.load_audio_file(audio_file, cfg.preprocess.sample_rate)
        ffmpeg.write_audio_file(audio, "moin.wav", cfg.preprocess.sample_rate)
        print(f"Audio loaded with duration {duration} and {num_channels} channels")

    audio_sample = load_and_process_audio(input_file, target_sampling_rate=24000)
    audio_sample = audio_sample[:24000*10]
    write_wav(audio_sample, 24000, "test.wav")

    model = EncodecModel.from_pretrained("facebook/encodec_24khz")
    processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
    print("The processor sampling rate is: ", processor.sampling_rate)
    
    # pre-process the inputs
    inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")

    encoder_outputs = model.encode(inputs['input_values'], bandwidth=24.0)

    # The model quantizer expects the audio codes to be of shape (quantizer_dim, batch_size, time_steps)
    audio_codes = encoder_outputs.audio_codes.view(1, encoder_outputs.audio_codes.size(2), encoder_outputs.audio_codes.size(3))
    audio_codes = rearrange(audio_codes, 'b q t -> q b t')
    embeddings = model.quantizer.decode(audio_codes)
    audio_values = model.decoder(embeddings)
    write_wav(audio_values[0].detach().numpy(), 24000, "test_reconstructed_from_embedding.wav")

    # Feedforward the audio codes to the decoder to get the audio values
    audio_values_direct = model.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales).audio_values
    write_wav(audio_values_direct[0].detach().numpy(), 24000, "test_reconstructed_directly.wav")
    exit()

    # torch.save({
    #     'X_ordered_training': X_ordered_training,
    #     'y_ordered_training': y_ordered_training,
    #     'X_ordered_testing': X_ordered_testing,
    #     'y_ordered_testing': y_ordered_testing
    # }, output_file_path)
    # print("Preprocessing done and data saved.")

if __name__ == "__main__":
    main()