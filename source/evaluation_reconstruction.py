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
    print("##### Starting Reconstruction Evaluation Stage #####")

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

    # Benchmarking for performance optimization
    if "cuda" in str(device):
        torch.backends.cudnn.benchmark = False # Disable cudnn benchmark for reproducibility, can lead to different algo choices
    # Make PyTorch operations deterministic for reproducibility
    if cfg.train.deterministic:
        # Why does it not work with warn_only=True for the Memory Efficient attention defaults to a non-deterministic algorithm?
        torch.use_deterministic_algorithms(mode=True, warn_only=False)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Set CUBLAS workspace config for reproducibility

    print(f"Torch deterministic algorithms: {torch.are_deterministic_algorithms_enabled()}")

    print(f"Creating the train dataset with db_path: {cfg.train.db_path_train}")
    train_dataset = MetaAudioDataset(db_path=cfg.train.db_path_train, max_num_samples=-1, has_audio=False, fast_forward_keygen=True)
    print(f"Creating the test dataset with db_path: {cfg.train.db_path_test}")
    test_dataset = MetaAudioDataset(db_path=cfg.train.db_path_test, max_num_samples=-1, has_audio=False, fast_forward_keygen=True)

    sampler_train = CustomSampler(dataset=train_dataset, pitch=cfg.train.pitch, shuffle=False, max_inst_per_family=cfg.train.max_inst_per_family, velocity=cfg.train.velocity)
    sampler_test = CustomSampler(dataset=test_dataset, pitch=cfg.train.pitch, shuffle=False, max_inst_per_family=cfg.train.max_inst_per_family, velocity=cfg.train.velocity)


    print(f"Creating the train dataloader with batch_size: {cfg.train.transformer.batch_size}")
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.train.transformer.batch_size,
                                  sampler=sampler_train,
                                  drop_last=False,
                                  num_workers=cfg.train.num_workers,
                                  )

    print(f"Creating the test dataloader with batch_size: {cfg.train.transformer.batch_size}")
    test_dataloader = DataLoader(test_dataset,
                                  batch_size=cfg.train.transformer.batch_size,
                                  sampler=sampler_test,
                                  drop_last=False,
                                  num_workers=cfg.train.num_workers,
                                )

    print(f"Length of train dataloader: {len(train_dataloader)}")
    print(f"Length of test dataloader: {len(test_dataloader)}")

    note_remap_tensor = torch.as_tensor(cfg.train.pitch, device=device)

    train_instrument_remap = np.unique(sampler_train.chosen_instruments)

    print(f"Number of instruments in train dataset: {len(train_instrument_remap)}")

    print(f"Creating the encodec model with model_name: {cfg.preprocess.model_name}")
    encodec_model = EncodecModel.from_pretrained(cfg.preprocess.model_name).to(device)
    encodec_model.eval()

    batch_size = 32
    loss_fn = F.mse_loss

    # number of tokens to compare
    num_tokens = 150

    # load vae
    print(f"Loading the VAE model from: {cfg.train.vae_path}")
    vae = torch.load(cfg.train.vae_path, map_location=device, weights_only=False)
    vae.device = device
    vae.eval()

    # # load transformer
    print(f"Loading the transformer model from: {cfg.train.transformer_path}")
    transformer = torch.load(cfg.train.transformer_path, weights_only=False).to(device)
    transformer.eval()

    def evaluate(dataloader):
        losses = []
        for b, data in enumerate(tqdm.tqdm(dataloader)):
            emb = data["embeddings"].to(device)

            # vae forward pass
            vae_output = vae.forward(emb)
            vae_predicted = vae_output[0]
            timbre_cond = vae_output[1].detach()
            pitch_cond = data['metadata']['pitch'].to(device)
            pitch_cond = (pitch_cond.unsqueeze(-1) == note_remap_tensor).nonzero(as_tuple=False)[..., -1]
            pitch_cond = torch.nn.functional.one_hot(pitch_cond, num_classes=len(note_remap_tensor)).float()
            
            # forward pass of transformer with combined conditioning
            combined_cond = torch.cat((timbre_cond, pitch_cond), dim=1)
            generated_transformer = transformer.generate(num_tokens=num_tokens, condition=combined_cond)

            # calculate reconstruction loss for vae and transformer
            loss_vae = loss_fn(emb[:,:num_tokens,:], vae_predicted[:,:num_tokens,:]).item()
            loss_transformer = loss_fn(emb[:,:num_tokens,:], generated_transformer[:,:num_tokens,:]).item()

            # fed the generated embeddings in the vae encoder again for pitch classification
            gt_pitch = data["metadata"]['pitch']
            gt_pitch = (gt_pitch.unsqueeze(-1) == note_remap_tensor.cpu()).nonzero(as_tuple=False)[..., -1].cpu().numpy()
            gt_note_cls_predicted = vae.forward(emb, encoder_only=True)[2].argmax(dim=1).cpu().numpy()
            vae_cls_predicted = vae.forward(vae_predicted, encoder_only=True)[2].argmax(dim=1).cpu().numpy()
            transformer_cls_predicted = vae.forward(generated_transformer, encoder_only=True)[2].argmax(dim=1).cpu().numpy()

            gt_pitch_01 = gt_pitch == gt_note_cls_predicted
            gt_pitch_01 = np.count_nonzero(gt_pitch_01) / len(gt_pitch_01)
            vae_pitch_01 = gt_pitch == vae_cls_predicted
            vae_pitch_01 = np.count_nonzero(vae_pitch_01) / len(vae_pitch_01)
            transformer_pitch_01 = gt_pitch == transformer_cls_predicted
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

    print("######## Evaluation ########")
    print("Evaluating on train dataset")
    evaluate(train_dataloader)
    print("Evaluating on test dataset")
    evaluate(test_dataloader)
