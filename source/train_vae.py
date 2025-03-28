import utils.debug

import os
from pathlib import Path

from dataset import MetaAudioDataset
from dataset import FilterPitchSampler

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from functools import partial

import torch
import torchinfo
from utils import logs, config

from hydra.utils import instantiate

from vae import ConditionConvVAE
import vae

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from transformers import EncodecModel

import tqdm

import utils.ffmpeg_helper as ffmpeg

import numpy as np
import matplotlib.pyplot as plt

LOG_TENSORBOARD = True

@torch.no_grad()
def eval_model(model, dl, device, max_num_batches, loss_fn, input_crop, current_epoch):
    model.eval()
    losses = []
    cls_pred = []
    cls_gt = []
    for i, data in enumerate(dl):
        emb = data['embeddings'].to(device)
        emb_pred, mean, var, note_cls, one_hot = model.forward(emb)
        gt_cls = data['metadata']['pitch'].to(device)
                
        rec_loss, reg_loss, cls_loss = loss_fn(x = emb[:,:input_crop,:], 
                                                            x_hat = emb_pred, 
                                                            mean = mean, 
                                                            var = var, 
                                                            note_cls = note_cls, 
                                                            gt_cls = gt_cls, 
                                                            current_epoch = current_epoch,
                                                            )

    
        
        
        
        losses.append([rec_loss.item(), reg_loss.item(), cls_loss.item()])
        
        cls_pred.extend(note_cls.argmax(dim=1).cpu().numpy())
        cls_gt.extend(gt_cls.cpu().numpy())
        
        if i >= max_num_batches: # data set can be very large
            break

    b = np.array(cls_pred) == np.array(cls_gt)
    acc01 = np.count_nonzero(b) / len(b)

    rec_loss, reg_loss, cls_loss = np.array(losses).mean(axis=0)
    return rec_loss, reg_loss, cls_loss, acc01

@torch.no_grad()
def visu_model(model, dl, device, input_crop, num_examples, name_prefix='', epoch=0, writer=None):
    model.eval()
    embs = np.zeros((0,input_crop,128))
    embs_pred = np.zeros((0,input_crop,128))
    means = np.zeros((0,2))
    families = np.zeros((0))
    for i, data in enumerate(dl):
        emb = data['embeddings'][:num_examples].to(device)
        emb_pred, mean, var, note_cls, one_hot  = model.forward(emb)
        embs = np.vstack((embs,emb[:,:input_crop,:].cpu().detach().numpy()))
        embs_pred = np.vstack((embs_pred,emb_pred.cpu().detach().numpy()))
        means = np.vstack((means, mean.cpu().detach().numpy()))
        families = np.concat((families, data['metadata']['family'].numpy()))
        if len(embs) >= num_examples: # skip when ds gets too large
            break

    
    # plot original embedding and decoded embedding
    fig1, axes1 = plt.subplots(1, 2, figsize=(10, 5), num=1)  # 1 row, 2 columns
    # orig embedding
    axes1[0].imshow(embs[0], vmin=0.0, vmax=1.0)
    axes1[0].set_title("Original")
    # generated embedding
    axes1[1].imshow(embs_pred[0], vmin=0.0, vmax=1.0)
    axes1[1].set_title("Generated")
    plt.savefig('out/vae/%s_embedding_comparison.png' % (name_prefix))

    # plot the latent as scatters
    fig2 = plt.figure(2)
    plt.scatter(means[:,0], means[:,1], c=families)
    plt.savefig('out/vae/%s_latent_visualization.png' % (name_prefix))

    if writer is not None:
        writer.add_figure(f"{name_prefix}/latent_visualization", fig2, epoch)
        writer.add_figure(f"{name_prefix}/embedding_comparison", fig1, epoch)

    plt.close(1)  # Schließt die bestehende Figur mit Nummer 1
    plt.close(2)
    fig1.clear()
    fig2.clear()

@torch.no_grad()
def hear_model(model, encodec_model, data_loader, device, input_crop, num_examples, name_prefix='', epoch=0, writer=None):
    model.eval()
    embs_decoded = []
    for i, data in enumerate(data_loader):
        emb = data['embeddings'][:num_examples].to(device)
        emb_pred, mean, var, note_cls, one_hot = model.forward(emb)

        emb_pred = MetaAudioDataset.denormalize_embedding(emb_pred)
        emb_pred = emb_pred.permute(0,2,1)
        decoded_pred = encodec_model.decoder(emb_pred)
        decoded_pred = decoded_pred.detach().cpu().numpy()

        for element in range(decoded_pred.shape[0]):
            embs_decoded.append(decoded_pred[element])
            if len(embs_decoded) >= num_examples:
                break
        if len(embs_decoded) >= num_examples:
            break

    for i, decoded in enumerate(embs_decoded):
        decoded_int = np.array(decoded * (2**15 - 1), dtype=np.int16)
        ffmpeg.write_audio_file(decoded_int, f"out/vae/{name_prefix}_generated_{i}.wav", 24000)
        if writer is not None:
            writer.add_audio(f"{name_prefix}/generated_{i}", decoded, epoch, sample_rate=24000)

def main():
    print("##### Starting Train Stage #####")
    os.makedirs("out/vae/checkpoints", exist_ok=True)

    # Check how many CUDA GPUs are available
    gpu_count = torch.cuda.device_count()
    print(f"Number of available GPUs: {gpu_count}")

    # Print GPU details if available
    if gpu_count > 0:
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Load the parameters from the dictionary into variables
    cfg = OmegaConf.load("params.yaml")

    epochs = cfg.train.vae.epochs

    # change learning rate at several points during training
    lr_change_intervals = [0.5, 0.6, 0.7, 0.8, 0.9]
    lr_change_multiplier = 0.5

    # Set a random seed for reproducibility across all devices. Add more devices if needed
    config.set_random_seeds(cfg.train.random_seed)
    # Prepare the requested device for training. Use cpu if the requested device is not available 
    device = config.prepare_device(cfg.train.device)

    print(f"Creating the train dataset with db_path: {cfg.train.db_path_train}")
    train_dataset = MetaAudioDataset(db_path=cfg.train.db_path_train, max_num_samples=-1, has_audio=False)
    # train_dataset = MetaAudioDataset(db_path=cfg.train.db_path_train, max_num_samples=1000) # for testing
    # train_dataset = MetaAudioDataset(db_path="data/partial/train_stripped", max_num_samples=1000, has_audio=False) # no audio data in the dataset
    print(f"Creating the valid dataset with db_path: {cfg.train.db_path_valid}")
    valid_dataset = MetaAudioDataset(db_path=cfg.train.db_path_valid, max_num_samples=-1, has_audio=False)
    # filter_pitch_sampler = FilterPitchSampler(dataset=valid_dataset, pitch=cfg.train.pitch)

    print(f"Creating the train dataloader with batch_size: {cfg.train.vae.batch_size}")
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.train.vae.batch_size,
                                  # sampler=FilterPitchSampler(valid_dataset, cfg.train.pitch, True),
                                  drop_last=False,
                                  num_workers=cfg.train.num_workers,
                                  shuffle=True,
                                  )

    print(f"Creating the valid dataloader with batch_size: {cfg.train.vae.batch_size}")
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=cfg.train.vae.batch_size,
                                  # sampler=FilterPitchSampler(valid_dataset, cfg.train.pitch, False),
                                  drop_last=False,
                                  num_workers=cfg.train.num_workers,
                                  shuffle=False,
                                )
    
    print('Size of train set: %d \t Size of val set: %d' % (len(train_dataset),len(valid_dataset)))
    
    print(f"Creating the vae model with channels: {cfg.train.vae.channels}, linears: {cfg.train.vae.linears}, input_crop: {cfg.train.vae.input_crop}")
    vae = ConditionConvVAE(cfg.train.vae.channels, cfg.train.vae.linears, cfg.train.vae.input_crop, device=device, dropout_ratio=cfg.train.vae.dropout_ratio, num_notes=128)

    print(f"Creating optimizer with lr: {cfg.train.vae.lr}, wd: {cfg.train.vae.wd}, betas: {cfg.train.vae.betas}")
    optimizer = torch.optim.AdamW(vae.parameters(), lr=cfg.train.vae.lr, weight_decay=cfg.train.vae.wd, betas=cfg.train.vae.betas)
    print("Instantiating the loss functions.")
    
    calculate_vae_loss = instantiate(cfg.train.vae.calculate_vae_loss, _recursive_=True)
    calculate_vae_loss = partial(calculate_vae_loss, device=device)

    encodec_model = None
    if cfg.train.vae.hear_interval > 0:
        print(f"Creating the encodec model with model_name: {cfg.preprocess.model_name}")
        encodec_model = EncodecModel.from_pretrained(cfg.preprocess.model_name).to(device)
        encodec_model.eval()

    writer = None
    if LOG_TENSORBOARD:
        tensorboard_path = logs.return_tensorboard_path()
        path_parts = Path(tensorboard_path).parts
        tensorboard_path = str(Path(*path_parts[:-1]) / "vae" / path_parts[-1])
        remote_dir = Path("logs/tensorboard/vae")
        remote_dir = logs.construct_remote_dir(remote_dir)
        metrics = {'train/reconstruction_loss': None,
                   'train/regularisation_loss': None,
                   'train/classifier_loss': None, 
                   'train/classifier_accuracy': None,
                   'valid/reconstruction_loss': None,
                   'valid/regularisation_loss': None,
                   'valid/classifier_loss': None,
                   'valid/classifier_accuracy': None}

        writer = logs.CustomSummaryWriter(log_dir=tensorboard_path, params=cfg, metrics=metrics, remote_dir=remote_dir)

        sample_inputs = torch.randn(1, 300, 128)
        vae.eval()
        writer.add_graph(vae, sample_inputs.to(device))

    print("######## Training ########")
    for epoch in range(epochs+1):
        #
        # training epoch
        #
        vae.train()
        for i, data in enumerate(tqdm.tqdm(train_dataloader)):
            optimizer.zero_grad()
            
            emb = data['embeddings'].to(device)
            emb_pred, mean, var, note_cls, one_hot = vae.forward(emb)
            
            rec_loss, reg_loss, cls_loss = calculate_vae_loss(x = emb[:,:cfg.train.vae.input_crop,:], 
                                                              x_hat = emb_pred, 
                                                              mean = mean, 
                                                              var = var, 
                                                              note_cls = note_cls, 
                                                              gt_cls = data['metadata']['pitch'].to(device), 
                                                              current_epoch = epoch,
                                                              )
            
            loss = rec_loss + reg_loss + cls_loss
            loss.backward()
            optimizer.step()


        # adapt learning rate
        if epoch in map(lambda x: int(epochs*x), lr_change_intervals):
            # adapt learning rate with multiplier
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_change_multiplier
            print('changed learning rate to %.3e after epoch %d' % (optimizer.param_groups[0]['lr'], epoch))
        
        #
        # evaluate model
        # 
        vae.eval()
        if (epoch % cfg.train.vae.eval_interval) == 0 and epoch > 0:
            # reconstruction error on validation dataset
            print()
                        
            losses = eval_model(model = vae, 
                                dl = train_dataloader, 
                                device = device, 
                                max_num_batches = 25, 
                                loss_fn = calculate_vae_loss, 
                                input_crop = cfg.train.vae.input_crop, 
                                current_epoch=epoch)
            print("TRAIN: Epoch %d: Reconstruction loss: %.6f, Regularization Loss: %.6f, Classifier Loss: %.6f, Classifier 0/1 Accuracy: %.6f" % (epoch, losses[0], losses[1], losses[2], losses[3]))
            if writer is not None:
                writer.add_scalar("train/reconstruction_loss", losses[0], epoch)
                writer.add_scalar("train/regularisation_loss", losses[1], epoch)
                writer.add_scalar("train/classifier_loss", losses[2], epoch)
                writer.add_scalar("train/classifier_accuracy", losses[3], epoch)

            losses = eval_model(model = vae, 
                                dl = valid_dataloader, 
                                device = device, 
                                max_num_batches = 25, 
                                loss_fn = calculate_vae_loss, 
                                input_crop = cfg.train.vae.input_crop, 
                                current_epoch=epoch)
            print("VAL: Epoch %d: Reconstruction loss: %.6f, Regularization Loss: %.6f, Classifier Loss: %.6f, Classifier 0/1 Accuracy: %.6f" % (epoch, losses[0], losses[1], losses[2], losses[3]))
            if writer is not None:
                writer.add_scalar("valid/reconstruction_loss", losses[0], epoch)
                writer.add_scalar("valid/regularisation_loss", losses[1], epoch)
                writer.add_scalar("valid/classifier_loss", losses[2], epoch)
                writer.add_scalar("valid/classifier_accuracy", losses[3], epoch)

        if (epoch % cfg.train.vae.visualize_interval) == 0 and epoch > 0:
            visu_model(vae, train_dataloader, device, cfg.train.vae.input_crop, name_prefix='train', num_examples=500, epoch=epoch, writer=writer)
            visu_model(vae, valid_dataloader, device, cfg.train.vae.input_crop, name_prefix='valid', num_examples=500, epoch=epoch, writer=writer)

        if cfg.train.vae.hear_interval > 0:
            if (epoch % cfg.train.vae.hear_interval) == 0 and epoch > 0:
                hear_model(vae, encodec_model, train_dataloader, device, cfg.train.vae.input_crop, 5, name_prefix='train', epoch=epoch, writer=writer)
                hear_model(vae, encodec_model, valid_dataloader, device, cfg.train.vae.input_crop, 5, name_prefix='valid', epoch=epoch, writer=writer)

        if (epoch % cfg.train.vae.save_interval) == 0 and epoch > 0:
            print("Saving model at epoch %d" % (epoch))
            torch.save(vae, 'out/vae/checkpoints/vae_epoch_%d.torch' % (epoch))

        if writer is not None:
            writer.step()
            
    print("Training completed. Saving the model.")
    torch.save(vae, 'out/vae/checkpoints/vae_final_epoch_%d.torch' % (epochs))
    if writer is not None:
        writer.close()
    
if __name__ == "__main__":
    main()
