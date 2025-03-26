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

@torch.no_grad()
def eval_model(model, cond_model, dl, device, num_batches):
    losses = []
    for b, data in enumerate(dl):
        dy = data["embeddings"].to(device)
        dy = dy.squeeze().swapaxes(1,2) # shape batch, time, features

        vae_output = cond_model.forward(dy)
        timbre_cond = vae_output[1].detach()
        pitch_cond = vae_output[4].detach()

        # concatenating timbre and pitch condition for putting into encoder of transformer
        combined_cond = torch.cat((timbre_cond, pitch_cond), dim=1)
        dx = dy[:,:-1,:]
        dx = torch.cat((torch.zeros((dx.shape[0],1,128)).to(device),dx),dim=1).detach()

        logits = model.forward(xdec=dx, xenc=combined_cond)
        loss = F.mse_loss(logits,dx).item()

        losses.append(loss)
        
        # calculate the mean abs error for a generated embedding (and not only the train loss)
        # takes a lot of time!
        gen_crop = 150
        generated = model.generate(gen_crop, combined_cond)
        gen_loss = F.mse_loss(dx[:,:gen_crop,:], generated[:,:gen_crop,:]).item()

        # losses.append(gen_loss)
        
        if b > num_batches:
            break
        
    return np.mean(losses).item()

def train():
    # Load data
    print("##### Starting Train Stage #####")
    os.makedirs("out/checkpoints", exist_ok=True)

    cfg = OmegaConf.load("params.yaml")
    
    # Set a random seed for reproducibility across all devices. Add more devices if needed
    config.set_random_seeds(cfg.train.random_seed)
    # Prepare the requested device for training. Use cpu if the requested device is not available 
    device = config.auto_device()


    epochs = 2000 # num passes through the dataset

    learning_rate = 1e-4 # max learning rate
    weight_decay = 0.05
    beta1 = 0.9
    beta2 = 0.95
    
    
    batch_size = 64

    # change learning rate at several points during training
    lr_change_intervals = [0.5, 0.6, 0.7, 0.8, 0.9]
    lr_change_multiplier = 0.5



    eval_num_samples = 5000
    eval_epoch = 200

    
    eval_epochs = []
    eval_train_losses = []
    eval_val_losses = []



    transformer_config = dict(
        block_size = 300,
        input_dimension = 128,
        internal_dimension = 512,
        feedforward_dimension = 2048,
        n_layer_encoder = 8,
        n_layer_decoder = 11,
        n_head = 8,
        dropout = 0.0
    )

    print(f"Creating the valid dataset and dataloader with db_path: {cfg.train.db_path_valid}")
    train_dataset = MetaAudioDataset(db_path=cfg.train.db_path_valid, max_num_samples=2, has_audio=False)
    # train_dataset = MetaAudioDataset(db_path=cfg.train.db_path_train, max_num_samples=1000) # for testing
    # train_dataset = MetaAudioDataset(db_path="data/partial/train_stripped", max_num_samples=1000, has_audio=False) # no audio data in the dataset
    valid_dataset = MetaAudioDataset(db_path=cfg.train.db_path_test, max_num_samples=2, has_audio=False)
    # filter_pitch_sampler = FilterPitchSampler(dataset=valid_dataset, pitch=cfg.train.pitch)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  # sampler=FilterPitchSampler(valid_dataset, cfg.train.pitch, True),
                                  drop_last=False,
                                  num_workers=cfg.train.num_workers,
                                  shuffle=False,
                                  )

    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=batch_size,
                                  # sampler=FilterPitchSampler(valid_dataset, cfg.train.pitch, False),
                                  drop_last=False,
                                  num_workers=cfg.train.num_workers,
                                  shuffle=False,
                                )
    
    
    model = GesamTransformer(transformer_config=transformer_config, device=device)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1,beta2))
    
    encodec_model = EncodecModel.from_pretrained(cfg.preprocess.model_name).to(device)
    condition_model = torch.load('out/checkpoints/vae.torch', weights_only=False).to(device)
    condition_model.eval()

    combined_cond_for_generation = torch.Tensor()
    for e in range(epochs):
        model.train()
        print('start epoch %d' % e)

        for _, data in enumerate(tqdm.tqdm(train_dataloader)):
            # autoregressive loss transformer trainingi
            optimizer.zero_grad()
            
            dy = data["embeddings"].to(device)
            dy = dy.squeeze().swapaxes(1,2) # shape batch, time, features

            vae_output = condition_model.forward(dy)
            timbre_cond = vae_output[1].detach()
            pitch_cond = vae_output[4].detach()

            # apply noise to condition vector for smoothing the output distribution
            # timbre_cond += torch.randn_like(timbre_cond).to(device) * torch.exp(0.5*vae_output[2])
            # pitch_cond += torch.randn_like(pitch_cond).to(device) * 0.05

            # concatenating timbre and pitch condition for putting into encoder of transformer
            combined_cond = torch.cat((timbre_cond, pitch_cond), dim=1)
            combined_cond_for_generation = combined_cond

            # dx is the transposed input vector (in time dimension) for autoregressive training
            dx = dy[:,:-1,:]
            dx = torch.cat((torch.zeros((dx.shape[0],1,128)).to(device),dx),dim=1).detach()
            
            logits = model.forward(xdec=dx, xenc=combined_cond)
            loss = F.mse_loss(logits,dy)
            
            loss.backward()
            optimizer.step()
            # print(loss)
            
        # adapt learning rate
        if e in map(lambda x: int(epochs*x), lr_change_intervals):
            # adapt learning rate with multiplier
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_change_multiplier
            print('changed learning rate to %.3e after epoch %d' % (optimizer.param_groups[0]['lr'], e))

        if (e + 1) % eval_epoch == 0 and e > 0:
            model.eval()
            print('evaluate model after epoch %d' % (e,))

            train_loss = eval_model(model, condition_model, train_dataloader, device=device, num_batches=eval_num_samples/batch_size)
            val_loss = eval_model(model, condition_model, valid_dataloader, device=device, num_batches=eval_num_samples/batch_size)
            
            print('######## train loss: %.6f (epoch %d)' % (train_loss, e))
            print('######## val loss: %.6f (epoch %d)' % (val_loss, e))
            print()

            # if len(eval_val_losses) > 0 and val_loss < min(eval_val_losses): 
            #     print('save new best model')
            #     torch.save(model, 'out/checkpoints/transformer.torch')
            
            eval_epochs.append(e)
            eval_train_losses.append(train_loss)
            eval_val_losses.append(val_loss)
            
            
            plt.close(0)
            plt.figure(0)
            plt.plot(eval_epochs, eval_train_losses, marker='o', linestyle='--', label='train')
            plt.plot(eval_epochs, eval_val_losses, marker='o', linestyle='--', label='val')
            plt.legend()
            plt.title('Train and Validation Losses after epoch %d' % e)
            plt.savefig('out/trainsformer_losses.png')


            num_generate = 2
            
            
            # print('generate %d random samples' % (num_generate,))
            # # generate condition vector combining pitch and timbre
            # timbre = torch.rand((num_generate, 2)) * 2.0 - 1.0
            # pitches = torch.zeros((num_generate,128))
            # for i in range(num_generate):
            #     ri = np.random.randint(21,100)
            #     pitches[i,ri] = 1.0
            # combined_cond = torch.cat((timbre, pitches), dim=1).to(device)



            generated = model.generate(300, combined_cond_for_generation)
            
            plt.close()
            plt.imshow(generated[0].cpu(), vmin=0.0, vmax=1.0)
            plt.savefig('out/transformer_generated.png')          
            plt.close()  
            plt.imshow(dy[0].cpu(), vmin=0.0, vmax=1.0)
            plt.savefig('out/transformer_gt.png')      
            emb_pred_for_audio = MetaAudioDataset.denormalize_embedding(generated)
            decoded = encodec_model.decoder((emb_pred_for_audio).permute(0,2,1))
            decoded = decoded.detach().cpu().numpy()
            decoded_int = np.int16(decoded * (2**15 - 1))
            for ind in range(num_generate):
                decoded_sample = decoded_int[ind]
                ffmpeg.write_audio_file(decoded_sample, "out/transformer_generated_%d.wav" % (ind,), 24000)


if __name__ == '__main__':
    train()