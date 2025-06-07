import random
import numpy as np
import gc
import matplotlib.pyplot as plt
import os
import shutil

import torch

import tqdm

from transformers import EncodecModel

from omegaconf import OmegaConf
from utils import logs, config
from hydra.utils import instantiate

from dataset import MetaAudioDataset

import utils.ffmpeg_helper as ffmpeg

def set_random_seeds(random_seed: int) -> None:
    if "random" in globals():
        random.seed(random_seed)  # type: ignore
    else:
        print("The 'random' package is not imported, skipping random seed.")

    if "np" in globals():
        np.random.seed(random_seed)  # type: ignore
    else:
        print("The 'numpy' package is not imported, skipping numpy seed.")

    if "torch" in globals():
        torch.manual_seed(random_seed)  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(random_seed)
    else:
        print("The 'torch' package is not imported, skipping torch seed.")
    if "scipy" in globals():
        scipy.random.seed(random_seed)  # type: ignore
    else:
        print("The 'scipy' package is not imported, skipping scipy seed.")

if __name__ == "__main__":

    out_dir = 'out/generate_audio_final/'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cfg = OmegaConf.load("params.yaml")

    # Prepare the requested device for training. Use cpu if the requested device is not available
    gpu_count = torch.cuda.device_count()
    print(f"Number of available GPUs: {gpu_count}")
    if gpu_count > 0:
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    device = config.prepare_device(cfg.train.device)

    # Set a random seed for reproducibility across typical libraries
    set_random_seeds(cfg.train.random_seed)

    encodec_model = EncodecModel.from_pretrained(cfg.preprocess.model_name).to(device)

    transformer = torch.load(cfg.train.transformer_path, weights_only=False).to(device)
    transformer.eval()

    note_remap_tensor = torch.as_tensor(cfg.train.pitch, device=device)

    sample_resolution = 10
    margin = (2.0/sample_resolution) / 2.
    sample_points = np.linspace(-1+margin,1-margin,sample_resolution)

    for pt in tqdm.tqdm(range(min(note_remap_tensor), max(note_remap_tensor) + 1)):
        for xi, xx in enumerate(sample_points):
            for yi, yy in enumerate(sample_points):
                current_filename = os.path.join(out_dir, '%03d_%03d_%03d.wav' % (pt, xi, yi))
                
                timbre_cond = torch.zeros((1, 2))
                timbre_cond[0,0] = xx
                timbre_cond[0,1] = yy
                pitch_cond = torch.zeros((1, len(note_remap_tensor)))
                pitch_index = (note_remap_tensor == pt).nonzero(as_tuple=True)[0].item()
                pitch_cond[0, pitch_index] = 1.0
                combined_cond = torch.cat((timbre_cond, pitch_cond), dim=1).to(device)

                generated = transformer.generate(300, combined_cond)
                
                generated = MetaAudioDataset.denormalize_embedding(generated)
                generated = generated.permute(0,2,1)
                decoded = encodec_model.decoder(generated)
                decoded = decoded.detach().cpu().numpy()

                decoded_max = np.max(np.abs(decoded))
                norm_factor = 0.8
                if decoded_max > norm_factor:
                    print(f"Warning: Decoded audio exceeds max amplitude of {norm_factor} ({decoded_max}), for {current_filename}.")

                # Normalize the decoded audio to the range [-norm_factor, norm_factor]
                decoded = (decoded / decoded_max) * norm_factor

                decoded_int = np.array(decoded * (2**15 - 1), dtype=np.int16)
                ffmpeg.write_audio_file(decoded_int, current_filename, 24000)

        