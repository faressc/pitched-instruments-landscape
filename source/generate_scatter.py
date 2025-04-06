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

from dataset import MetaAudioDataset, CustomSampler

import shutil

out_dir = 'out/generate_scatter/'
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir, exist_ok=True)

cfg = OmegaConf.load("params.yaml")

# Set a random seed for reproducibility across all devices. Add more devices if needed
config.set_random_seeds(cfg.train.random_seed)
# Prepare the requested device for training. Use cpu if the requested device is not available 
device = config.prepare_device(cfg.train.device)
batch_size = 32



print(f"Creating the valid dataset and dataloader with db_path: {cfg.train.db_path_valid}")
train_dataset = MetaAudioDataset(db_path=cfg.train.db_path_valid, max_num_samples=-1, has_audio=False)
sampler_train = CustomSampler(dataset=train_dataset, pitch=cfg.train.pitch, shuffle=True)
dl = DataLoader(train_dataset,
                batch_size=batch_size,
                sampler=sampler_train,
                drop_last=False,
                # num_workers=cfg.train.num_workers,
                )


encodec_model = EncodecModel.from_pretrained(cfg.preprocess.model_name).to(device)


vae = torch.load(cfg.train.vae_path, map_location=device, weights_only=False)
vae.device = device
vae.eval()

pitch_timbres = dict()
for i in range(128):
    pitch_timbres[i] = np.zeros((0,3))

for s in tqdm.tqdm(dl):
    _, timbre_embedding, _, _, pitch_classification, cls_head = vae.forward(s['embeddings'].to(device))
    pitch_classification = pitch_classification.argmax(dim=1)
    pitch_classification = pitch_classification.cpu().detach().numpy()
    timbre_embedding = timbre_embedding.cpu().detach().numpy()
    family = s['metadata']['family'].numpy()
    # family = s['metadata']['instrument'].numpy()
    
    for t,p,f in zip(timbre_embedding, pitch_classification, family):
        pitch_timbres[p] = np.vstack((pitch_timbres[p], np.hstack((t,f))))
        


pitches = cfg.train.pitch

for pt in range(min(pitches), max(pitches)+1):
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

    plt.savefig(os.path.join(out_dir,'%03d.svg' % (pt,)))
    plt.close()
