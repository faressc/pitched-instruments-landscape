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

print(f"Creating the train dataset and dataloader with db_path: {cfg.train.db_path_train}")
train_dataset = MetaAudioDataset(db_path=cfg.train.db_path_train, max_num_samples=-1, has_audio=False, fast_forward_keygen=True)
sampler_train = CustomSampler(dataset=train_dataset, pitch=cfg.train.pitch, velocity=cfg.train.velocity, max_inst_per_family=cfg.train.max_inst_per_family, shuffle=False)
dl = DataLoader(train_dataset,
                batch_size=cfg.train.vae.batch_size,
                sampler=sampler_train,
                drop_last=False,
                num_workers=cfg.train.num_workers,
                )


encodec_model = EncodecModel.from_pretrained(cfg.preprocess.model_name).to(device)

vae = torch.load(cfg.train.vae_path, map_location=device, weights_only=False)
vae.device = device
vae.eval()

pitch_timbres = dict()
instrument_ids = dict()
for i in range(128):
    pitch_timbres[i] = np.zeros((0,3))
    instrument_ids[i] = []

for s in tqdm.tqdm(dl):
    _, timbre_embedding, _, _, pitch_classification, family_cls = vae.forward(s['embeddings'].to(device))
    pitch_classification = pitch_classification.argmax(dim=1)
    pitch_classification = pitch_classification.cpu().detach().numpy()
    timbre_embedding = timbre_embedding.cpu().detach().numpy()
    family = s['metadata']['family'].numpy()
    instrument_id = s['metadata']['instrument_str']
    
    for t,p,f,id in zip(timbre_embedding, pitch_classification, family, instrument_id):
        pitch_timbres[p] = np.vstack((pitch_timbres[p], np.hstack((t,f))))
        instrument_ids[p].append(id)

family_colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan', 'magenta', 'yellow']

for pt in range(min(cfg.train.pitch), max(cfg.train.pitch)+1):
    print('pitch %d: %d number of points' % (pt, len(pitch_timbres[pt])))
    # Create the scatter plot
    fig, ax = plt.subplots(1, figsize=(6, 6))

    # Create a scatter plot with each point having its own instrument label
    # Convert family indices to integer type for indexing
    family_indices = pitch_timbres[pt][:,2].astype(int)
    
    # Get colors from family_colors list based on family indices
    point_colors = [family_colors[idx % len(family_colors)] for idx in family_indices]
    
    for i in range(len(instrument_ids[pt])):
        ax.scatter(pitch_timbres[pt][i,0], pitch_timbres[pt][i,1],
                   c=point_colors[i],
                   edgecolor='k',
                #    label=instrument_ids[pt][i],
                   s=50)
        
    # ax.legend(loc='upper right', fontsize=8)

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
