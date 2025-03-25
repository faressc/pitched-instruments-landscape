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



def train():
    cfg = OmegaConf.load("params.yaml")
    
    # Set a random seed for reproducibility across all devices. Add more devices if needed
    config.set_random_seeds(cfg.train.random_seed)
    # Prepare the requested device for training. Use cpu if the requested device is not available 

    # Load the parameters from the dictionary into variables
    cfg = OmegaConf.load("params.yaml")
    

    # Set a random seed for reproducibility across all devices. Add more devices if needed
    config.set_random_seeds(cfg.train.random_seed)
    # Prepare the requested device for training. Use cpu if the requested device is not available 
    device = config.auto_device()

    print(f"Creating the valid dataset and dataloader with db_path: {cfg.train.db_path_valid}")
    train_dataset = MetaAudioDataset(db_path=cfg.train.db_path_train, has_audio=False)
    
    pitches = np.zeros((128,))
    
    for d in train_dataset:
        pitches[d['metadata']['pitch']] += 1
        
    plt.plot(pitches)
    plt.savefig('out/pitch_distribution.png')
    


if __name__ == '__main__':
    train()