import utils.debug

import os
from pathlib import Path

from dataset import MetaAudioDataset, CustomSampler

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
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg (non-interactive)
import matplotlib.pyplot as plt

LOG_TENSORBOARD = True

@torch.no_grad()
def eval_model(model, dl, device, max_num_batches, loss_fn, input_crop, current_epoch, note_remap, instrument_remap):
    model.eval()
    losses = []
    note_cls_pred = []
    note_cls_gt = []
    family_cls_pred = []
    family_cls_gt = []
    instrument_cls_pred = []
    instrument_cls_gt = []
    for i, data in enumerate(dl):
        emb = data['embeddings'].to(device)
        emb_pred, mean, logvar, note_cls, one_hot, family_cls, instrument_cls = model.forward(emb)
        gt_note_cls = data['metadata']['pitch'].to(device)
        gt_inst = data['metadata']['instrument'].to(device)

        rec_loss, rep_loss, spa_loss, kl_loss, note_cls_loss, family_cls_loss, instrument_cls_loss = loss_fn(x = emb[:,:input_crop,:], 
                                                            x_hat = emb_pred, 
                                                            mean = mean, 
                                                            logvar = logvar, 
                                                            note_cls = note_cls, 
                                                            gt_note_cls = gt_note_cls, 
                                                            family_cls = family_cls,
                                                            gt_family = data['metadata']['family'].to(device), 
                                                            instrument_cls = instrument_cls,
                                                            gt_inst = gt_inst, 
                                                            current_epoch = current_epoch,
                                                            note_remap = note_remap,
                                                            instrument_remap = instrument_remap)

        losses.append([rec_loss.item(), rep_loss.item(), spa_loss.item(), kl_loss.item(), note_cls_loss.item(), family_cls_loss.item(), instrument_cls_loss.item()])

        # remap note and instrument classes
        if note_remap is not None:
            # Ensure note_remap is a tensor on the correct device
            note_remap_tensor = torch.as_tensor(note_remap, device=gt_note_cls.device)
            # For each value in gt_note_cls, find its index in note_remap
            gt_note_cls = (gt_note_cls.unsqueeze(-1) == note_remap_tensor).nonzero(as_tuple=False)[..., -1]
        
        if instrument_remap is not None:
            # Ensure instrument_remap is a tensor on the correct device
            instrument_remap_tensor = torch.as_tensor(instrument_remap, device=gt_inst.device)
        else:
            instrument_remap_tensor = torch.unique(gt_inst)
            
        # For each value in gt_inst, find its index in instrument_remap
        gt_inst = (gt_inst.unsqueeze(-1) == instrument_remap_tensor).nonzero(as_tuple=False)[..., -1]

        note_cls_pred.extend(note_cls.argmax(dim=1).cpu().numpy())
        note_cls_gt.extend(gt_note_cls.cpu().numpy())

        family_cls_pred.extend(family_cls.argmax(dim=1).cpu().numpy())
        family_cls_gt.extend(data['metadata']['family'].cpu().numpy())

        instrument_cls_pred.extend(instrument_cls.argmax(dim=1).cpu().numpy())
        instrument_cls_gt.extend(gt_inst.cpu().numpy())

        if i >= max_num_batches: # data set can be very large
            break

    note_acc01 = np.array(note_cls_pred) == np.array(note_cls_gt)
    note_acc01 = np.count_nonzero(note_acc01) / len(note_acc01)

    family_acc01 = np.array(family_cls_pred) == np.array(family_cls_gt)
    family_acc01 = np.count_nonzero(family_acc01) / len(family_acc01)

    instrument_acc01 = np.array(instrument_cls_pred) == np.array(instrument_cls_gt)
    instrument_acc01 = np.count_nonzero(instrument_acc01) / len(instrument_acc01)

    rec_loss, rep_loss, spa_loss, kl_loss, note_cls_loss, family_cls_loss, instrument_cls_loss = np.array(losses).mean(axis=0)
    return rec_loss, rep_loss, spa_loss, kl_loss, note_cls_loss, family_cls_loss, instrument_cls_loss, note_acc01, family_acc01, instrument_acc01

@torch.no_grad()
def visu_model(model, dl, device, input_crop, num_examples, name_prefix='', epoch=0, writer=None, instrument_remap=None):
    model.eval()
    embs = np.zeros((0,input_crop,128))
    embs_pred = np.zeros((0,input_crop,128))
    means = np.zeros((0,2))
    families = np.zeros((0))
    instruments = np.zeros((0))
    for i, data in enumerate(dl):
        emb = data['embeddings'][:num_examples].to(device)
        emb_pred, mean, logvar, note_cls, one_hot, family_cls, instrument_cls = model.forward(emb)
        embs = np.vstack((embs,emb[:,:input_crop,:].cpu().detach().numpy()))
        embs_pred = np.vstack((embs_pred,emb_pred.cpu().detach().numpy()))
        means = np.vstack((means, mean.cpu().detach().numpy()))
        families = np.concat((families, data['metadata']['family'][:num_examples].numpy()))
        instruments = np.concat((instruments, data['metadata']['instrument'][:num_examples].numpy()))
        if len(embs) >= num_examples: # skip when ds gets too large
            break

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
    if instrument_remap is not None:
        # Remap instruments to appropriate indices
        instrument_remap_tensor = np.asarray(instrument_remap)
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
    
    # Create colormap for the scatter plot
    from matplotlib import colormaps
    cmap = colormaps['hsv']  # Use HSV colormap for better differentiation

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
    fig2 = plt.figure(2, figsize=(10, 8))
    scatter = plt.scatter(means[:,0], means[:,1], c=instrument_colors, cmap=cmap)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    
    # Add a colorbar to show the mapping from color to family
    cbar = plt.colorbar(scatter)
    cbar.set_label('Instrument Family')
    
    plt.title(f"Latent Space Visualization - {name_prefix}")
    plt.savefig(f'out/vae/{name_prefix}_latent_visualization.png')

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
        emb_pred, mean, logvar, note_cls, one_hot, family_cls, instrument_cls = model.forward(emb)

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


def main():
    print("##### Starting VAE Train Stage #####")

    print(f"DVC Experiment Name: {os.getenv('DVC_EXP_NAME')}")

    os.makedirs("out/vae/checkpoints", exist_ok=True)
    
    # Load the parameters from the dictionary into variables
    cfg = OmegaConf.load("params.yaml")

    epochs = cfg.train.vae.epochs

    # change learning rate at several points during training
    lr_change_intervals = [0.5, 0.6, 0.7, 0.8, 0.9]
    lr_change_multiplier = 0.5

    # Prepare the requested device for training. Use cpu if the requested device is not available
    gpu_count = torch.cuda.device_count()
    print(f"Number of available GPUs: {gpu_count}")
    if gpu_count > 0:
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    device = config.prepare_device(cfg.train.device)

    # Set a random seed for reproducibility across typical libraries
    set_random_seeds(cfg.train.random_seed)
    
    # Benchmarking for performance optimization
    if "cuda" in str(device):
        torch.backends.cudnn.benchmark = False # Disable cudnn benchmark for reproducibility, can lead to different algo choices
    # Make PyTorch operations deterministic for reproducibility
    if cfg.train.deterministic:
        torch.use_deterministic_algorithms(mode=True, warn_only=True)
    print(f"Torch deterministic algorithms: {torch.are_deterministic_algorithms_enabled()}")

    print(f"Creating the train dataset with db_path: {cfg.train.db_path_train}")
    train_dataset = MetaAudioDataset(db_path=cfg.train.db_path_valid, max_num_samples=-1, has_audio=False, fast_forward_keygen=True)

    print(f"Creating the valid dataset with db_path: {cfg.train.db_path_valid}")
    valid_dataset = MetaAudioDataset(db_path=cfg.train.db_path_valid, max_num_samples=-1, has_audio=False, fast_forward_keygen=True)
    
    sampler_train = CustomSampler(dataset=train_dataset, pitch=cfg.train.pitch, max_inst_per_family=cfg.train.max_inst_per_family, velocity=cfg.train.velocity, shuffle=True)
    sampler_valid = CustomSampler(dataset=valid_dataset, pitch=cfg.train.pitch, max_inst_per_family=cfg.train.max_inst_per_family, velocity=cfg.train.velocity, shuffle=False)

    print(f"Creating the train dataloader with batch_size: {cfg.train.vae.batch_size}")
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.train.vae.batch_size,
                                  sampler=sampler_train,
                                  drop_last=False,
                                  num_workers=cfg.train.num_workers,
                                  )

    print(f"Creating the valid dataloader with batch_size: {cfg.train.vae.batch_size}")
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=cfg.train.vae.batch_size,
                                  sampler=sampler_valid,
                                  drop_last=False,
                                  num_workers=cfg.train.num_workers,
                                  )
    
    print(f"Length of train dataloader: {len(train_dataloader)}")
    print(f"Length of valid dataloader: {len(valid_dataloader)}")

    train_instrument_remap = np.unique(sampler_train.chosen_instruments)
    valid_instrument_remap = np.unique(sampler_valid.chosen_instruments)

    print(f"Number of instruments in train dataset: {len(train_instrument_remap)}")
    print(f"Number of instruments in valid dataset: {len(valid_instrument_remap)}")

    print(f"Creating the vae model with channels: {cfg.train.vae.channels}, linears: {cfg.train.vae.linears}, input_crop: {cfg.train.vae.input_crop}")
    vae = ConditionConvVAE(cfg.train.vae.channels, cfg.train.vae.linears, cfg.train.vae.input_crop, device=device, dropout_ratio=cfg.train.vae.dropout_ratio, num_notes=len(cfg.train.pitch), num_instruments=len(train_instrument_remap))

    print(f"Creating optimizer with lr: {cfg.train.vae.lr}, wd: {cfg.train.vae.wd}, betas: {cfg.train.vae.betas}")
    optimizer = torch.optim.AdamW(vae.parameters(), lr=cfg.train.vae.lr, weight_decay=cfg.train.vae.wd, betas=cfg.train.vae.betas)
    print("Instantiating the loss functions.")
    
    calculate_vae_loss = instantiate(cfg.train.vae.calculate_vae_loss, _recursive_=True)
    calculate_vae_loss = partial(calculate_vae_loss, device=device, rec_beta = cfg.train.vae.rec_beta, neighbor_beta = cfg.train.vae.neighbor_beta, spa_beta = cfg.train.vae.spa_beta, kl_beta = cfg.train.vae.kl_beta, note_cls_beta = cfg.train.vae.note_cls_beta, family_cls_beta = cfg.train.vae.family_cls_beta, instrument_cls_beta = cfg.train.vae.instrument_cls_beta, reg_scaling_exp_neighbor = cfg.train.vae.reg_scaling_exp_neighbor, reg_scaling_exp_family = cfg.train.vae.reg_scaling_exp_family, reg_scaling_exp_instrument = cfg.train.vae.reg_scaling_exp_instrument, note_remap=cfg.train.pitch)

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
                  'train/repulsion_loss': None,
                  'train/spatial_loss': None,
                  'train/kl_loss': None,
                  'train/note_classifier_loss': None,
                  'train/note_classifier_accuracy': None,
                  'train/family_classifier_loss': None,
                  'train/family_classifier_accuracy': None,
                  'train/instrument_classifier_loss': None,
                  'train/instrument_classifier_accuracy': None,
                  'valid/reconstruction_loss': None,
                  'valid/repulsion_loss': None,
                  'valid/spatial_loss': None,
                  'valid/kl_loss': None,
                  'valid/note_classifier_loss': None,
                  'valid/note_classifier_accuracy': None,
                  'valid/family_classifier_loss': None,
                  'valid/family_classifier_accuracy': None,}
                  # Instruments are different in train and valid dataset, so classifier metrics for validation make no sense

        writer = logs.CustomSummaryWriter(log_dir=tensorboard_path, params=cfg, metrics=metrics, sync_interval=cfg.train.vae.eval_interval, remote_dir=remote_dir)

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
            emb_pred, mean, logvar, note_cls, one_hot, family_cls, instrument_cls = vae.forward(emb)

            rec_loss, rep_loss, spa_loss, kl_loss, note_cls_loss, family_cls_loss, instrument_cls_loss = calculate_vae_loss(x = emb[:,:cfg.train.vae.input_crop,:],
                                                              x_hat = emb_pred, 
                                                              mean = mean, 
                                                              logvar = logvar, 
                                                              note_cls = note_cls, 
                                                              gt_note_cls = data['metadata']['pitch'].to(device), 
                                                              family_cls = family_cls,
                                                              gt_family = data['metadata']['family'].to(device),
                                                              instrument_cls = instrument_cls,
                                                              gt_inst = data['metadata']['instrument'].to(device), 
                                                              current_epoch = epoch,
                                                              instrument_remap=train_instrument_remap)

            loss = rec_loss + rep_loss + spa_loss + kl_loss + note_cls_loss + family_cls_loss + instrument_cls_loss
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

            rec_loss, rep_loss, spa_loss, kl_loss, note_cls_loss, family_cls_loss, instrument_cls_loss, note_acc01, family_acc01, instrument_acc01 = eval_model(model = vae,
                                dl = train_dataloader,
                                device = device,
                                max_num_batches = 25,
                                loss_fn = calculate_vae_loss,
                                input_crop = cfg.train.vae.input_crop,
                                current_epoch=epoch,
                                note_remap=cfg.train.pitch,
                                instrument_remap=train_instrument_remap)
            print("TRAIN: Epoch %d: Reconstruction loss: %.6f, Repulsion Loss: %.6f, Spatial Loss: %.6f, KL Loss: %.6f, Pitch Classifier Loss: %.6f, Pitch 0/1 Accuracy: %.6f, Family Classifier Loss: %.6f, Family 0/1 Accuracy: %.6f, Instrument Classifier Loss: %.6f, Instrument 0/1 Accuracy: %.6f" % (epoch, rec_loss, rep_loss, spa_loss, kl_loss, note_cls_loss, note_acc01, family_cls_loss, family_acc01, instrument_cls_loss, instrument_acc01))
            if writer is not None:
                writer.add_scalar("train/reconstruction_loss", rec_loss, epoch)
                writer.add_scalar("train/repulsion_loss", rep_loss, epoch)
                writer.add_scalar("train/spatial_loss", spa_loss, epoch)
                writer.add_scalar("train/kl_loss", kl_loss, epoch)
                writer.add_scalar("train/note_classifier_loss", note_cls_loss, epoch)
                writer.add_scalar("train/note_classifier_accuracy", note_acc01, epoch)
                writer.add_scalar("train/family_classifier_loss", family_cls_loss, epoch)
                writer.add_scalar("train/family_classifier_accuracy", family_acc01, epoch)
                writer.add_scalar("train/instrument_classifier_loss", instrument_cls_loss, epoch)
                writer.add_scalar("train/instrument_classifier_accuracy", instrument_acc01, epoch)

            rec_loss, rep_loss, spa_loss, kl_loss, note_cls_loss, family_cls_loss, instrument_cls_loss, note_acc01, family_acc01, instrument_acc01 = eval_model(model = vae,
                                dl = valid_dataloader,
                                device = device,
                                max_num_batches = 25,
                                loss_fn = calculate_vae_loss,
                                input_crop = cfg.train.vae.input_crop,
                                current_epoch=epoch,
                                note_remap=cfg.train.pitch,
                                instrument_remap=None)
            print("VALID: Epoch %d: Reconstruction loss: %.6f, Repulsion Loss: %.6f, Spatial Loss: %.6f, KL Loss: %.6f, Pitch Classifier Loss: %.6f, Pitch 0/1 Accuracy: %.6f, Family Classifier Loss: %.6f, Family 0/1 Accuracy: %.6f" % (epoch, rec_loss, rep_loss, spa_loss, kl_loss, note_cls_loss, note_acc01, family_cls_loss, family_acc01))
            if writer is not None:
                writer.add_scalar("valid/reconstruction_loss", rec_loss, epoch)
                writer.add_scalar("valid/repulsion_loss", rep_loss, epoch)
                writer.add_scalar("valid/spatial_loss", spa_loss, epoch)
                writer.add_scalar("valid/kl_loss", kl_loss, epoch)
                writer.add_scalar("valid/note_classifier_loss", note_cls_loss, epoch)
                writer.add_scalar("valid/note_classifier_accuracy", note_acc01, epoch)
                writer.add_scalar("valid/family_classifier_loss", family_cls_loss, epoch)
                writer.add_scalar("valid/family_classifier_accuracy", family_acc01, epoch)

        if (epoch % cfg.train.vae.visualize_interval) == 0 and epoch > 0:
            visu_model(vae, train_dataloader, device, cfg.train.vae.input_crop, name_prefix='train', num_examples=500, epoch=epoch, writer=writer, instrument_remap=train_instrument_remap)
            visu_model(vae, valid_dataloader, device, cfg.train.vae.input_crop, name_prefix='valid', num_examples=500, epoch=epoch, writer=writer, instrument_remap=None)

        if cfg.train.vae.hear_interval > 0:
            if (epoch % cfg.train.vae.hear_interval) == 0 and epoch > 0:
                hear_model(vae, encodec_model, train_dataloader, device, cfg.train.vae.input_crop, 5, name_prefix='train', epoch=epoch, writer=writer)
                hear_model(vae, encodec_model, valid_dataloader, device, cfg.train.vae.input_crop, 5, name_prefix='valid', epoch=epoch, writer=writer)

        if (epoch % cfg.train.vae.save_interval) == 0 and epoch > 0:
            print("Saving model at epoch %d" % (epoch))
            torch.save(vae, 'out/vae/checkpoints/vae_epoch_%d.torch' % (epoch))

        if writer is not None and epoch > 0:
            writer.step()
            
    print("Training completed. Saving the model.")
    torch.save(vae, f'out/vae/checkpoints/vae_final_{os.getenv("DVC_EXP_NAME")}_epoch_{epochs}.torch')
    if writer is not None:
        writer.close()
    
if __name__ == "__main__":
    main()
