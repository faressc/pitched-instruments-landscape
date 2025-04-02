import random
import numpy as np
import gc
import matplotlib.pyplot as plt
import os
import shutil

import torch

from transformers import EncodecModel

from omegaconf import OmegaConf
from utils import logs, config
from hydra.utils import instantiate

from dataset import MetaAudioDataset

import utils.ffmpeg_helper as ffmpeg


out_dir = 'out/generate_audio/'
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir, exist_ok=True)
    

cfg = OmegaConf.load("params.yaml")

# Set a random seed for reproducibility across all devices. Add more devices if needed
config.set_random_seeds(cfg.train.random_seed)
# Prepare the requested device for training. Use cpu if the requested device is not available 
device = config.prepare_device(cfg.train.device)
batch_size = 32


encodec_model = EncodecModel.from_pretrained(cfg.preprocess.model_name).to(device)

transformer = torch.load(cfg.train.transformer_path, weights_only=False).to(device)
transformer.eval()

pitches = cfg.train.pitch


sample_resolution = 5
margin = (2.0/sample_resolution) / 2.
sample_points = np.linspace(-1+margin,1-margin,sample_resolution)


for pt in range(min(pitches), max(pitches)+1):
    for xi, xx in enumerate(sample_points):
        for yi, yy in enumerate(sample_points):
            current_filename = os.path.join(out_dir, '%03d_%03d_%03d.wav' % (pt, xi, yi))
            print('Generate: %s' % (current_filename,))
            
            timbre_cond = torch.zeros((1, 2))
            timbre_cond[0,0] = xx
            timbre_cond[0,1] = yy
            pitch_cond = torch.zeros((1,128))
            pitch_cond[0, pt] = 1.0
            combined_cond = torch.cat((timbre_cond, pitch_cond), dim=1).to(device)

            generated = transformer.generate(300, combined_cond)
            
            generated = MetaAudioDataset.denormalize_embedding(generated)
            generated = generated.permute(0,2,1)
            decoded = encodec_model.decoder(generated)
            decoded = decoded.detach().cpu().numpy()

            decoded_int = np.array(decoded * (2**15 - 1), dtype=np.int16)
            ffmpeg.write_audio_file(decoded_int, current_filename, 24000)

        