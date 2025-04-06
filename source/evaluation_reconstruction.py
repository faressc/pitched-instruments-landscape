import random
import numpy as np
import gc
import matplotlib.pyplot as plt
import os
import shutil

import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import EncodecModel

from omegaconf import OmegaConf
from utils import logs, config
from hydra.utils import instantiate

from dataset import MetaAudioDataset, CustomSampler

import utils.ffmpeg_helper as ffmpeg


cfg = OmegaConf.load("params.yaml")

# Set a random seed for reproducibility across all devices. Add more devices if needed
config.set_random_seeds(cfg.train.random_seed)
# Prepare the requested device for training. Use cpu if the requested device is not available 
device = config.prepare_device(cfg.train.device)
batch_size = 32
loss_fn = F.mse_loss

# number of samples to evaluate before aborting (for big train set tests)
max_num_samples = 50000
# number of tokens to compare
num_tokens = 150


# load dataset (train / test)
ds = MetaAudioDataset(db_path=cfg.train.db_path_train, max_num_samples=max_num_samples, has_audio=False)
sampler = CustomSampler(dataset=ds, pitch=cfg.train.pitch, velocity=cfg.train.velocity, shuffle=True)
dl = DataLoader(ds,
                batch_size=batch_size,
                sampler=sampler,
                drop_last=False,
                )


# load encodec
encodec_model = EncodecModel.from_pretrained(cfg.preprocess.model_name).to(device)


# load vae
vae_old_path = 'out/vae_old/checkpoints/vae_final_epoch_1000.torch'
vae = torch.load(cfg.train.vae_path, map_location=device, weights_only=False)
vae.device = device
vae.eval()
vae_old = torch.load(vae_old_path, map_location=device, weights_only=False)
vae_old.device = device
vae_old.eval()


# # load transformer
transformer = torch.load(cfg.train.transformer_path, weights_only=False).to(device)
transformer.eval()

losses = []
for b, data in enumerate(tqdm.tqdm(dl)):
    emb = data["embeddings"].to(device)

    vae_old_output = vae_old.forward(emb)
    vae_output = vae.forward(emb)

    # concatenating timbre and pitch condition for putting into encoder of transformer
    timbre_cond = vae_old_output[1].detach()
    pitch_cond = vae_old_output[4].detach()
    combined_cond = torch.cat((timbre_cond, pitch_cond), dim=1)
    emb_shifted = emb[:,:-1,:]
    emb_shifted = torch.cat((torch.zeros((emb_shifted.shape[0],1,128)).to(device), emb_shifted),dim=1).detach()

    # Calculate the loss the way we did in training with triangular mask
    logits = transformer.forward(xdec=emb_shifted, xenc=combined_cond)
    generated_transformer = transformer.generate(num_tokens=num_tokens, condition=combined_cond)

    
    loss_vae = loss_fn(emb[:,:num_tokens,:], vae_output[0][:,:num_tokens,:]).item()
    loss_transformer = loss_fn(emb[:,:num_tokens,:], generated_transformer[:,:num_tokens,:]).item()

    losses.append([loss_vae, loss_transformer])


mean_loss_vae, mean_loss_transformer = np.mean(losses, axis=0)
print('mean loss vae: %.6f \t mean loss transformer: %.6f' % (mean_loss_vae, mean_loss_transformer))
