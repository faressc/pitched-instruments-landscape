import utils.debug

import os
from pathlib import Path

from dataset import MetaAudioDataset, CustomSampler

from utils import logs, config
import utils.ffmpeg_helper as ffmpeg
from transformer import GesamTransformer

from omegaconf import OmegaConf
from hydra.utils import instantiate

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import EncodecModel

import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg (non-interactive)
import matplotlib.pyplot as plt

def main():
    print("##### Starting Train Stage #####")

    cfg = OmegaConf.load("params.yaml")
    
    # Prepare the requested device for training. Use cpu if the requested device is not available
    gpu_count = torch.cuda.device_count()
    print(f"Number of available GPUs: {gpu_count}")
    if gpu_count > 0:
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    device = config.prepare_device(cfg.train.device)

    # Set a random seed for reproducibility across typical libraries
    config.set_random_seeds(cfg.train.random_seed)
    # Benchmarking for performance optimization
    if "cuda" in str(device):
        torch.backends.cudnn.benchmark = True # TODO: Does this work with deterministic algorithms?
    # Make PyTorch operations deterministic for reproducibility
    if cfg.train.deterministic:
        torch.use_deterministic_algorithms(mode=True, warn_only=True)
    print(f"Torch deterministic algorithms: {torch.are_deterministic_algorithms_enabled()}")

    print(f"Creating the train dataset with db_path: {cfg.train.db_path_train}")
    train_dataset = MetaAudioDataset(db_path=cfg.train.db_path_train, max_num_samples=-1, has_audio=False, fast_forward_keygen=False)

    print(f"Creating the test dataset with db_path: {cfg.train.db_path_test}")
    test_dataset = MetaAudioDataset(db_path=cfg.train.db_path_valid, max_num_samples=-1, has_audio=False, fast_forward_keygen=False)
    
    sampler_train = CustomSampler(dataset=train_dataset, pitch=cfg.train.pitch, max_inst_per_family=-1, velocity=[100], shuffle=False)
    sampler_test = CustomSampler(dataset=test_dataset, pitch=cfg.train.pitch, max_inst_per_family=-1, velocity=[100], shuffle=False)

    print(f"Creating the train dataloader with batch_size: {cfg.train.vae.batch_size}")
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.train.vae.batch_size,
                                  sampler=sampler_train,
                                  drop_last=False,
                                  num_workers=cfg.train.num_workers,
                                  )

    print(f"Creating the test dataloader with batch_size: {cfg.train.vae.batch_size}")
    test_dataloader = DataLoader(test_dataset,
                                  batch_size=cfg.train.vae.batch_size,
                                  sampler=sampler_test,
                                  drop_last=False,
                                  num_workers=cfg.train.num_workers,
                                  )
    
    print(f"Length of train dataloader: {len(train_dataloader)}")
    print(f"Length of test dataloader: {len(test_dataloader)}")
    
    print(f"Loading the condition model from path: {cfg.train.vae_path}")
    condition_model = torch.load(cfg.train.vae_path, map_location=device, weights_only=False)
    condition_model.device = device
    condition_model.eval()

    print("######## Evaluation ########")
    print("Evaluating the condition model")

    # Function to calculate variance statistics for a dataloader
    def calculate_variance_stats(dataloader, name):
        pitches_per_instrument = dict()
        instrument_ids_per_pitch = dict()

        print(f"\nProcessing {name} data...")
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            emb = data["embeddings"].to(device)
            inst_id = data["metadata"]["instrument"]
            pitch = data["metadata"]["pitch"]

            vae_output = condition_model.forward(emb)
            timbre_cond = vae_output[1].detach().cpu().numpy()
            pitch_cond = vae_output[4].detach().cpu().numpy()

            pitch_info = np.hstack((timbre_cond, np.expand_dims(pitch, axis=1)))

            for i, inst in enumerate(inst_id):
                inst = int(inst)
                pitches_per_instrument.setdefault(inst, np.zeros(shape=(0,3)))
                pitches_per_instrument[inst] = np.vstack((pitches_per_instrument[inst], pitch_info[i]))

            instrument_info = np.hstack((timbre_cond, np.expand_dims(inst_id, axis=1)))
            for i, p in enumerate(pitch):
                p = int(p)
                instrument_ids_per_pitch.setdefault(p, np.zeros(shape=(0,3)))
                instrument_ids_per_pitch[p] = np.vstack((instrument_ids_per_pitch[p], instrument_info[i]))

        # Calculate the variance of the pitch for each instrument
        pitch_variance_per_instrument = dict()
        print(f"\n{name} - Pitch variance per instrument:")
        for inst, pitch_info in pitches_per_instrument.items():
            pitch_variance_per_instrument[inst] = np.var(pitch_info[:, :2], axis=0)
            print(f"Instrument {inst} pitch variance: {pitch_variance_per_instrument[inst]}")

        instrument_variance_per_pitch = dict()
        print(f"\n{name} - Instrument variance per pitch:")
        for pitch, instrument_info in instrument_ids_per_pitch.items():
            instrument_variance_per_pitch[pitch] = np.var(instrument_info[:, :2], axis=0)
            print(f"Pitch {pitch} instrument variance: {instrument_variance_per_pitch[pitch]}")

        mean_variance_per_instrument = np.zeros(shape=(0,2))
        for inst, pitch_variance in pitch_variance_per_instrument.items():
            mean_variance_per_instrument = np.vstack((mean_variance_per_instrument, pitch_variance))
        print(f"\n{name} - Mean variance per instrument: {np.mean(mean_variance_per_instrument, axis=0)}")

        mean_variance_per_pitch = np.zeros(shape=(0,2))
        for pitch, instrument_variance in instrument_variance_per_pitch.items():
            mean_variance_per_pitch = np.vstack((mean_variance_per_pitch, instrument_variance))
        print(f"{name} - Mean variance per pitch: {np.mean(mean_variance_per_pitch, axis=0)}")
        
        return pitch_variance_per_instrument, instrument_variance_per_pitch

    # Calculate variance statistics for training data
    train_pitch_variance, train_instrument_variance = calculate_variance_stats(train_dataloader, "Train")
    
    # Calculate variance statistics for test data
    test_pitch_variance, test_instrument_variance = calculate_variance_stats(test_dataloader, "Test")


if __name__ == '__main__':
    main()