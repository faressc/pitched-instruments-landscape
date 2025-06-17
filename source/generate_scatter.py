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
try:
    from proto.meta_audio_file_pb2 import MetaAudioFile
except:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from proto.meta_audio_file_pb2 import MetaAudioFile

import shutil

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

    out_dir = 'out/generate_scatter_final/'

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

    print(f"Creating the train dataset with db_path: {cfg.train.db_path_train}")
    train_dataset = MetaAudioDataset(db_path=cfg.train.db_path_train, max_num_samples=-1, has_audio=False, fast_forward_keygen=True)

    sampler_train = CustomSampler(dataset=train_dataset, pitch=cfg.train.pitch, max_inst_per_family=cfg.train.max_inst_per_family, velocity=cfg.train.velocity, shuffle=True)

    print(f"Creating the train dataloader with batch_size: {cfg.train.vae.batch_size}")
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.train.vae.batch_size,
                                  sampler=sampler_train,
                                  drop_last=False,
                                  num_workers=cfg.train.num_workers,
                                  )
    
    print(f"Length of train dataloader: {len(train_dataloader)}")

    train_instrument_remap = np.unique(sampler_train.chosen_instruments)

    print(f"Number of instruments in train dataset: {len(train_instrument_remap)}")

    print(f"Creating the encodec model with model_name: {cfg.preprocess.model_name}")
    encodec_model = EncodecModel.from_pretrained(cfg.preprocess.model_name).to(device)
    encodec_model.eval()

    print(f"Loading the VAE model from: {cfg.train.vae_path}")
    model = torch.load(cfg.train.vae_path, map_location=device, weights_only=False)
    model.eval()

    input_crop = cfg.train.vae.input_crop

    pitch_timbres = dict()
    means = np.zeros((0,2))
    families = np.zeros((0))
    instruments = np.zeros((0))
    pitches = np.zeros((0))
    for data in tqdm.tqdm(train_dataloader, desc="Inference"):
        emb = data['embeddings'].to(device)
        _, mean, _, _, _, family_cls, instrument_cls = model.forward(emb)
        means = np.vstack((means, mean.cpu().detach().numpy()))
        families = np.concatenate((families, data['metadata']['family'].numpy()))
        instruments = np.concatenate((instruments, data['metadata']['instrument'].numpy()))
        pitches = np.concatenate((pitches, data['metadata']['pitch'].numpy()))
        for p in data['metadata']['pitch'].numpy():
            if pitch_timbres.get(p) is None:
                pitch_timbres[p] = np.zeros((0,3))

    # Create a mapping from instrument to family
    unique_instruments = np.unique(instruments)
    instrument_to_family = {}
    for i, inst in enumerate(unique_instruments):
        family_idx = families[instruments == inst][0]
        instrument_to_family[inst] = family_idx

    # Get unique families
    unique_families = np.unique(list(instrument_to_family.values()))
    family_to_color_base = {family: idx/len(unique_families) for idx, family in enumerate(unique_families)}
    
    # Create a color map that assigns similar colors to instruments in the same family
    instrument_colors = []
    if train_instrument_remap is not None:
        # Remap instruments to appropriate indices
        instrument_remap_tensor = np.asarray(train_instrument_remap)
        remapped_instruments = []
        for inst in instruments:
            idx = np.where(instrument_remap_tensor == inst)[0][0]
            remapped_instruments.append(idx)
            
            # Get the family for this instrument
            family = instrument_to_family[inst]
            # Get the base color for this family
            base_color = family_to_color_base[family]
            
            # Calculate a small offset for the instrument within its family
            family_instruments = [i for i, fam in instrument_to_family.items() if fam == family]
            offset = family_instruments.index(inst) / (len(family_instruments) + 1) * 0.1
            
            # Create a color with a slight variation from the family's base color
            instrument_colors.append(base_color + offset)
            
        instruments = np.array(remapped_instruments)
    else:
        # If no remap is provided, just use instrument indices directly
        for inst in instruments:
            family = instrument_to_family[inst]
            base_color = family_to_color_base[family]
            
            family_instruments = [i for i, fam in instrument_to_family.items() if fam == family]
            offset = family_instruments.index(inst) / (len(family_instruments) + 1) * 0.1
            
            instrument_colors.append(base_color + offset)

    for m, i, p in zip(means, instrument_colors, pitches):
        pitch_timbres[p] = np.vstack((pitch_timbres[p], np.hstack((m,i))))
    
    # Create colormap for the scatter plot
    from matplotlib import colormaps
    cmap = colormaps['hsv']  # Use HSV colormap for better differentiation

    pitches = pitches.astype(int)
    for pt in tqdm.tqdm(range(min(pitches), max(pitches) + 1), desc="Generating scatter plots"):

        # Create the scatter plot
        fig, ax = plt.subplots(1, figsize=(6, 6))
        scatter = plt.scatter(pitch_timbres[pt][:,0], pitch_timbres[pt][:,1], c=pitch_timbres[pt][:,2], cmap=cmap, edgecolor='k', s=50)

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

    # Generate a plot with all the pitches combined
    fig, ax = plt.subplots(1, figsize=(6, 6))
    all_values = np.vstack(list(pitch_timbres.values()))
    scatter = plt.scatter(all_values[:,0], all_values[:,1], c=all_values[:,2], cmap=cmap, edgecolor='k', s=50)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks(np.arange(-1, 1.5, 0.5))  # Include endpoints
    ax.set_yticks(np.arange(-1, 1.5, 0.5))
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.savefig(os.path.join(out_dir,'all_pitches.svg'))
    plt.close()

    # Plot only the colorbar (legend) for instrument families
    cmap = plt.get_cmap('hsv')
    family_colors = [cmap(family_to_color_base[family]) for family in unique_families]

    fig, ax = plt.subplots(figsize=(10, 0.6))

    # Create a ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])

    # Draw the colorbar only (no scatter plot)
    cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')

    # Set colorbar ticks at the base color positions (as floats between 0 and 1)
    tick_locs = [family_to_color_base[family] for family in unique_families]
    cbar.set_ticks(tick_locs)

    # Calculate positions for the labels (midway between ticks)
    padded_ticks = tick_locs + [1.0]
    label_locs = [(padded_ticks[i] + padded_ticks[i+1])/2 for i in range(len(padded_ticks)-1)]

    # Set the tick labels at the calculated midpoint positions
    tick_labels = [MetaAudioFile.Metadata.InstrumentFamily.Name(int(family)) for family in unique_families]
    tick_labels = [label.split('_')[0].lower().capitalize() for label in tick_labels]
    tick_labels = [label.replace('Keyboard', 'Key') for label in tick_labels]

    # Use the set_ticks and set_ticklabels separately to position labels between ticks
    cbar.ax.set_xticks(label_locs, minor=True)
    cbar.ax.set_xticklabels(tick_labels, minor=True)
    cbar.ax.set_xticklabels([], minor=False)  # Hide the major tick labels

    # Make minor ticks (our labels) more prominent
    cbar.ax.tick_params(axis='x', which='minor', length=0, labelsize=16)

    # Save the figure with tight layout
    plt.tight_layout(pad=0)  
    plt.savefig(os.path.join(out_dir, 'legend.svg'))

    # Add some padding around the figure to ensure labels aren't cut off
    fig.set_size_inches(10, 0.8)
    plt.tight_layout(pad=0.5)  
    plt.savefig(os.path.join(out_dir, 'legend_padded.svg'))

    plt.close()
