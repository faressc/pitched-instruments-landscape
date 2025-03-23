import random
import numpy as np
import gc
import matplotlib.pyplot as plt
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import EncodecModel

from omegaconf import OmegaConf
from utils import logs, config
from hydra.utils import instantiate

from dataset import MetaAudioDataset

from transformer import GesamTransformer


# calculate loss of model for a given dataset (executed during training)
@torch.no_grad()
def det_loss_testing(ds, model, condition_model):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False) # get new dataloader because we want random sampling here!
    losses = []
    gen_losses = []
    gen_loss_crop = 150
    for dx, dy, _, _ in dl:
        dx = dx.to(device)
        dy = dy.to(device)
        logits, loss = model.predict(condition_model, dx, dy)
        losses.append(loss.cpu().detach().item())    
        condition_bottleneck = condition_model(dx,True)[0].detach()
        gx = model.generate(gen_loss_crop, condition_bottleneck)
        gen_loss = torch.mean(torch.abs(dx[:,:gen_loss_crop,:]-gx[:,:gen_loss_crop,:])).item()
        gen_losses.append(gen_loss)
    return np.mean(losses), np.mean(gen_losses)


def train():
    cfg = OmegaConf.load("params.yaml")
    
    # Set a random seed for reproducibility across all devices. Add more devices if needed
    config.set_random_seeds(cfg.train.random_seed)
    # Prepare the requested device for training. Use cpu if the requested device is not available 
    device = config.auto_device()


    epochs = 500 # num passes through the dataset

    learning_rate = 9e-5 # max learning rate
    weight_decay = 0.05
    beta1 = 0.9
    beta2 = 0.95
    
    
    batch_size = 128

    # change learning rate at several points during training
    lr_change_intervals = [0.5, 0.6, 0.7, 0.8, 0.9]
    lr_change_multiplier = 0.5



    stats_every_iteration = 10
    train_set_testing_size = 1000
    eval_epoch = 2
    visu_epoch = 2


    transformer_config = dict(
        block_size = 300,
        block_size_encoder = 130,
        input_dimension = 128,
        internal_dimension = 512,
        feedforward_dimension = 2048,
        n_layer_encoder = 4,
        n_layer_decoder = 11,
        n_head = 8,
        dropout = 0.25
    )


    # Load data
    print("##### Starting Train Stage #####")
    os.makedirs("out/checkpoints", exist_ok=True)


    # Load the parameters from the dictionary into variables
    cfg = OmegaConf.load("params.yaml")
    

    # Set a random seed for reproducibility across all devices. Add more devices if needed
    config.set_random_seeds(cfg.train.random_seed)
    # Prepare the requested device for training. Use cpu if the requested device is not available 
    device = config.auto_device()

    print(f"Creating the valid dataset and dataloader with db_path: {cfg.train.db_path_valid}")
    train_dataset = MetaAudioDataset(db_path=cfg.train.db_path_train, has_audio=False)
    # train_dataset = MetaAudioDataset(db_path=cfg.train.db_path_train, max_num_samples=1000) # for testing
    # train_dataset = MetaAudioDataset(db_path="data/partial/train_stripped", max_num_samples=1000, has_audio=False) # no audio data in the dataset
    valid_dataset = MetaAudioDataset(db_path=cfg.train.db_path_valid, has_audio=False)
    # filter_pitch_sampler = FilterPitchSampler(dataset=valid_dataset, pitch=cfg.train.pitch)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  # sampler=FilterPitchSampler(valid_dataset, cfg.train.pitch, True),
                                  drop_last=False,
                                  num_workers=cfg.train.num_workers,
                                  shuffle=True,
                                  )

    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=batch_size,
                                  # sampler=FilterPitchSampler(valid_dataset, cfg.train.pitch, False),
                                  drop_last=False,
                                  num_workers=cfg.train.num_workers,
                                  shuffle=False,
                                )
    

    
    # do not change
    best_val_loss = 1e9
    best_val_loss_iter = 0
    best_train_loss = 1e9
    best_train_loss_iter = 0
    best_train_gen_loss = 1e9
    best_train_gen_loss_iter = 0
    best_val_gen_loss = 1e9
    best_val_gen_loss_iter = 0

    iterations = [] # for plotting
    train_losses = []
    val_losses = []
    train_gen_losses = []
    val_gen_losses = []
    change_learning_rate = learning_rate
    actual_learning_rate = learning_rate
    
    model = GesamTransformer(transformer_config=transformer_config, device=device)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1,beta2))
    
    encodec_model = EncodecModel.from_pretrained(cfg.preprocess.model_name).to(device)
    condition_model = torch.load('out/checkpoints/vae.torch', weights_only=False).to(device)
    condition_model.eval()

    for e in range(epochs):
        model.train()
        print('start iteration %d' % e)
        for _, data in enumerate(train_dataloader):
            # autoregressive loss transformer trainingi
            optimizer.zero_grad()
            
            y = data["embeddings"].to(device)
            y = y.squeeze().swapaxes(1,2) # shape batch, time, features
            # x is the transposed input vector (in time dimension) for autoregressive training
            x = y[:,:-1,:]
            x = torch.cat((torch.zeros((x.shape[0],1,128)).to(device),x),dim=1)

            _, loss = model.predict(condition_model, x, y)                
            loss.backward()
            optimizer.step()
            print(loss)
            
        print('epoch %d done' % e)

        # adapt learning rate
        if e in map(lambda x: int(epochs*x), lr_change_intervals):
            # adapt learning rate with multiplier
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_change_multiplier
            print('changed learning rate to %.3e after epoch %d' % (optimizer.param_groups[0]['lr'], e))



if __name__ == '__main__':
    train()