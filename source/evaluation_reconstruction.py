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
ds = MetaAudioDataset(db_path=cfg.train.db_path_test, max_num_samples=max_num_samples, has_audio=False)
sampler = CustomSampler(dataset=ds, pitch=cfg.train.pitch, velocity=cfg.train.velocity, max_inst_per_family=cfg.train.max_inst_per_family, shuffle=True)
dl = DataLoader(ds,
                batch_size=batch_size,
                sampler=sampler,
                drop_last=False,
                )


# load encodec
encodec_model = EncodecModel.from_pretrained(cfg.preprocess.model_name).to(device)


# load vae
vae = torch.load(cfg.train.vae_path, map_location=device, weights_only=False)
vae.device = device
vae.eval()

# # load transformer
transformer = torch.load(cfg.train.transformer_path, weights_only=False).to(device)
transformer.eval()

losses = []
for b, data in enumerate(tqdm.tqdm(dl)):
    emb = data["embeddings"].to(device)

    # vae forward pass
    vae_output = vae.forward(emb)
    vae_predicted = vae_output[0]
    timbre_cond = vae_output[1].detach()
    pitch_cond = vae_output[4].detach()
    
    # forward pass of transformer with combined conditioning
    combined_cond = torch.cat((timbre_cond, pitch_cond), dim=1)
    generated_transformer = transformer.generate(num_tokens=num_tokens, condition=combined_cond)

    # calculate reconstruction loss for vae and transformer
    loss_vae = loss_fn(emb[:,:num_tokens,:], vae_predicted[:,:num_tokens,:]).item()
    loss_transformer = loss_fn(emb[:,:num_tokens,:], generated_transformer[:,:num_tokens,:]).item()

    # fed the generated embeddings in the vae encoder again for pitch classification
    gt_pitch = data["metadata"]['pitch']
    gt_cls_predicted = vae.forward(emb, encoder_only=True)[2].argmax(dim=1).cpu().numpy()
    vae_cls_predicted = vae.forward(vae_predicted, encoder_only=True)[2].argmax(dim=1).cpu().numpy()
    transformer_cls_predicted = vae.forward(generated_transformer, encoder_only=True)[2].argmax(dim=1).cpu().numpy()

    
    gt_pitch_01 = np.array(gt_pitch) == np.array(gt_cls_predicted)
    gt_pitch_01 = np.count_nonzero(gt_pitch_01) / len(gt_pitch_01)
    vae_pitch_01 = np.array(gt_pitch) == np.array(vae_cls_predicted)
    vae_pitch_01 = np.count_nonzero(vae_pitch_01) / len(vae_pitch_01)
    transformer_pitch_01 = np.array(gt_pitch) == np.array(transformer_cls_predicted)
    transformer_pitch_01 = np.count_nonzero(transformer_pitch_01) / len(transformer_pitch_01)

    smp_ind = 1
    img_gt = emb[smp_ind,:num_tokens,:].cpu().detach().numpy()
    img_vae = vae_predicted[smp_ind].cpu().detach().numpy()
    img_transformer = generated_transformer[smp_ind].cpu().detach().numpy()

    save_crop = 30
    cmap = 'viridis'
    plt.imsave("out/img_gt.png", img_gt[:save_crop,:], cmap=cmap)
    plt.imsave("out/img_vae.png", img_vae[:save_crop,:], cmap=cmap)
    plt.imsave("out/img_transformer.png", img_transformer[:save_crop,:], cmap=cmap)


    losses.append([loss_vae, loss_transformer, gt_pitch_01, vae_pitch_01, transformer_pitch_01])


loss_vae, loss_transformer, gt_pitch_01, vae_pitch_01, transformer_pitch_01 = np.mean(losses, axis=0)
print('mean loss vae: %.6f \t mean loss transformer: %.6f' % (loss_vae, loss_transformer))
print('gt pitch classification 0/1 accuracy: %.6f \t vae pitch classification 0/1 accuracy: %.6f \t transformer pitch classification 0/1 accuracy: %.6f' % (gt_pitch_01, vae_pitch_01, transformer_pitch_01))

