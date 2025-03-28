import random
import numpy as np
import gc
import matplotlib.pyplot as plt
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import EncodecModel

import tqdm

from omegaconf import OmegaConf
from utils import logs, config
from hydra.utils import instantiate

from dataset import MetaAudioDataset

from transformer import GesamTransformer

import utils.ffmpeg_helper as ffmpeg


eval_dir = 'out/evaluate/'

print("##### Starting Train Stage #####")
os.makedirs(eval_dir, exist_ok=True)

cfg = OmegaConf.load("params.yaml")

# Set a random seed for reproducibility across all devices. Add more devices if needed
config.set_random_seeds(cfg.train.random_seed)
# Prepare the requested device for training. Use cpu if the requested device is not available 
device = config.auto_device()

print(f"Creating the valid dataset and dataloader with db_path: {cfg.train.db_path_valid}")
ds = MetaAudioDataset(db_path=cfg.train.db_path_valid, max_num_samples=-1, has_audio=False)

batch_size = 32




dl = DataLoader(ds,
                                batch_size=batch_size,
                                # sampler=FilterPitchSampler(valid_dataset, cfg.train.pitch, True),
                                drop_last=False,
                                # num_workers=cfg.train.num_workers,
                                shuffle=False,
                                )


encodec_model = EncodecModel.from_pretrained(cfg.preprocess.model_name).to(device)


vae = torch.load('out/vae/checkpoints/vae_final_epoch_300.torch', weights_only=False).to(device)
vae.eval()

transformer = torch.load('out/transformer/checkpoints/transformer_epoch_29.torch', weights_only=False).to(device)
transformer.eval()


pitch_timbres = dict()

for i in range(128):
    pitch_timbres[i] = np.zeros((0,3))



for s in tqdm.tqdm(dl):
    _, timbre_embedding, _, _, pitch_classification = vae.forward(s['embeddings'].to(device))
    pitch_classification = pitch_classification.argmax(dim=1)
    pitch_classification = pitch_classification.cpu().detach().numpy()
    timbre_embedding = timbre_embedding.cpu().detach().numpy()
    family = s['metadata']['family'].numpy()
    
    for t,p,f in zip(timbre_embedding, pitch_classification, family):
        pitch_timbres[p] = np.vstack((pitch_timbres[p], np.hstack((t,f))))
        


min_pitch = 21
max_pitch = 108

for pt in range(min_pitch, max_pitch+1):
    print('pitch %d: %d number of points' % (pt, len(pitch_timbres[pt])))
    
    
    # Create the scatter plot
    fig, ax = plt.subplots(1, figsize=(6, 6))
    scatter = plt.scatter(pitch_timbres[pt][:,0], pitch_timbres[pt][:,1], c=pitch_timbres[pt][:,2], cmap='viridis', edgecolor='k', s=50)

    # ax.axis('off')

    # Set axis limits and equal scale
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal', adjustable='box')

    # Adjust layout to remove padding around the plot
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Show the plot
    plt.grid(True)
    ax.set_xticks(np.arange(-1, 1.2, 0.2))  # Include endpoints
    ax.set_yticks(np.arange(-1, 1.2, 0.2))
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

    plt.savefig(os.path.join(eval_dir,'timbres_note_%03d.svg' % (pt,)))
