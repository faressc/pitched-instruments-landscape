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

LOG_TENSORBOARD = True

@torch.no_grad()
def eval_model(model, cond_model, dl, device, num_batches, loss_fn):
    model.eval()
    losses = []
    for b, data in enumerate(dl):
        emb = data["embeddings"].to(device)

        vae_output = cond_model.forward(emb)
        timbre_cond = vae_output[1].detach()
        pitch_cond = vae_output[4].detach()

        # concatenating timbre and pitch condition for putting into encoder of transformer
        combined_cond = torch.cat((timbre_cond, pitch_cond), dim=1)
        emb_shifted = emb[:,:-1,:]
        emb_shifted = torch.cat((torch.zeros((emb_shifted.shape[0],1,128)).to(device), emb_shifted),dim=1).detach()

        # Calculate the loss the way we did in training with triangular mask
        logits = model.forward(xdec=emb_shifted, xenc=combined_cond)
        loss = loss_fn(logits, emb).item()

        # Generate the whole sequence with the model
        num_tokens = 150
        generated = model.generate(num_tokens=num_tokens, condition=combined_cond)
        gen_loss = loss_fn(generated[:,:num_tokens,:], emb[:,:num_tokens,:]).item()

        losses.append([loss, gen_loss])
        
        if b >= num_batches:
            break

    loss, gen_loss = np.mean(losses, axis=0)
        
    return loss, gen_loss

@torch.no_grad()
def visu_model(model, cond_model, dl, device, name_prefix="", epoch=0, writer=None):
    model.eval()
    for i, data in enumerate(dl):
        emb = data["embeddings"].to(device)
        vae_output = cond_model.forward(emb)
        timbre_cond = vae_output[1].detach()
        pitch_cond = vae_output[4].detach()
        combined_cond = torch.cat((timbre_cond, pitch_cond), dim=1)

        generated = model.generate(emb.shape[1], combined_cond)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))
        # orig embedding
        axes[0].imshow(emb[0].cpu().detach().numpy(), vmin=0.0, vmax=1.0)
        axes[0].set_title("Original")
        # generated embedding
        axes[1].imshow(generated[0].cpu().detach().numpy(), vmin=0.0, vmax=1.0)
        axes[1].set_title("Generated")
        plt.savefig(f"out/transformer/{name_prefix}_embedding_comparison.png")

        if writer is not None:
            writer.add_figure(f"{name_prefix}/embedding_comparison", fig, epoch)

        plt.close()
        fig.clear()
        break

@torch.no_grad()
def hear_model(model, cond_model, encodec_model, dl, device, num_examples, name_prefix="", epoch=0, writer=None):
    model.eval()
    embs_decoded = []
    for i, data in enumerate(dl):
        emb = data["embeddings"][:num_examples].to(device)
        vae_output = cond_model.forward(emb)
        timbre_cond = vae_output[1].detach()
        pitch_cond = vae_output[4].detach()
        combined_cond = torch.cat((timbre_cond, pitch_cond), dim=1)

        generated = model.generate(emb.shape[1], combined_cond)
        generated = MetaAudioDataset.denormalize_embedding(generated)
        generated = generated.permute(0,2,1)
        decoded = encodec_model.decoder(generated)
        decoded = decoded.detach().cpu().numpy()

        for element in range(decoded.shape[0]):
            embs_decoded.append(decoded[element])
            if len(embs_decoded) >= num_examples:
                break
        if len(embs_decoded) >= num_examples:
            break

    for i, decoded in enumerate(embs_decoded):
        decoded_int = np.array(decoded * (2**15 - 1), dtype=np.int16)
        ffmpeg.write_audio_file(decoded_int, f"out/transformer/{name_prefix}_generated_{i}.wav", 24000)
        if writer is not None:
            writer.add_audio(f"{name_prefix}/generated_{i}", decoded, epoch, sample_rate=24000)

def main():
    print("##### Starting Train Stage #####")
    os.makedirs("out/transformer/checkpoints", exist_ok=True)

    cfg = OmegaConf.load("params.yaml")

    epochs = cfg.train.transformer.epochs

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
    config.set_random_seeds(cfg.train.random_seed)
    # Benchmarking for performance optimization
    if "cuda" in str(device):
        torch.backends.cudnn.benchmark = True # TODO: Does this work with deterministic algorithms?
    # Make PyTorch operations deterministic for reproducibility
    if cfg.train.deterministic:
        torch.use_deterministic_algorithms(mode=True, warn_only=True)
    print(f"Torch deterministic algorithms: {torch.are_deterministic_algorithms_enabled()}")

    print(f"Creating the train dataset with db_path: {cfg.train.db_path_train}")
    train_dataset = MetaAudioDataset(db_path=cfg.train.db_path_train, max_num_samples=-1, has_audio=False, fast_forward_keygen=True)
    print(f"Creating the valid dataset with db_path: {cfg.train.db_path_valid}")
    valid_dataset = MetaAudioDataset(db_path=cfg.train.db_path_valid, max_num_samples=-1, has_audio=False, fast_forward_keygen=True)

    sampler_train = CustomSampler(dataset=train_dataset, pitch=cfg.train.pitch, shuffle=True, max_inst_per_family=cfg.train.max_inst_per_family, velocity=cfg.train.velocity)
    sampler_valid = CustomSampler(dataset=valid_dataset, pitch=cfg.train.pitch, shuffle=False, max_inst_per_family=cfg.train.max_inst_per_family, velocity=cfg.train.velocity)


    print(f"Creating the train dataloader with batch_size: {cfg.train.transformer.batch_size}")
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.train.transformer.batch_size,
                                  sampler=sampler_train,
                                  drop_last=False,
                                  num_workers=cfg.train.num_workers,
                                  )

    print(f"Creating the valid dataloader with batch_size: {cfg.train.transformer.batch_size}")
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=cfg.train.transformer.batch_size,
                                  sampler=sampler_valid,
                                  drop_last=False,
                                  num_workers=cfg.train.num_workers,
                                )

    print(f"Length of train dataloader: {len(train_dataloader)}")
    print(f"Length of valid dataloader: {len(valid_dataloader)}")

    transformer_config = dict(
        block_size = cfg.train.transformer.block_size,
        input_dimension = cfg.train.transformer.input_dimension,
        internal_dimension = cfg.train.transformer.internal_dimension,
        feedforward_dimension = cfg.train.transformer.feedforward_dimension,
        n_layer_encoder = cfg.train.transformer.n_layer_encoder,
        n_layer_decoder = cfg.train.transformer.n_layer_decoder,
        n_head = cfg.train.transformer.n_head,
        dropout = cfg.train.transformer.dropout
    )

    print(f"Creating the transformer model with config: {transformer_config}")
    model = GesamTransformer(transformer_config=transformer_config, device=device)
    
    print(f"Creating the optimizer with learning_rate: {cfg.train.transformer.learning_rate}, weight_decay: {cfg.train.transformer.weight_decay}, betas: {cfg.train.transformer.betas}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.transformer.learning_rate, weight_decay=cfg.train.transformer.weight_decay, betas=cfg.train.transformer.betas)

    print("Instantiating the loss function")
    loss_fn = instantiate(cfg.train.transformer.loss_fn)
    
    print(f"Loading the condition model from path: {cfg.train.transformer.condition_model_path}")
    condition_model = torch.load(cfg.train.transformer.condition_model_path, map_location=device, weights_only=False)
    condition_model.device = device
    condition_model.eval()

    encodec_model = None
    if cfg.train.transformer.hear_interval > 0:
        print(f"Creating the encodec model with model_name: {cfg.preprocess.model_name}")
        encodec_model = EncodecModel.from_pretrained(cfg.preprocess.model_name).to(device)
        encodec_model.eval()

    writer = None
    if LOG_TENSORBOARD:
        tensorboard_path = logs.return_tensorboard_path()
        path_parts = Path(tensorboard_path).parts
        tensorboard_path = str(Path(*path_parts[:-1]) / "transformer" / path_parts[-1])
        remote_dir = Path("logs/tensorboard/transformer")
        remote_dir = logs.construct_remote_dir(remote_dir)
        metrics = {'train/loss': None,
                   'train/gen_loss': None,
                   'valid/loss': None,
                   'valid/gen_loss': None}

        writer = logs.CustomSummaryWriter(log_dir=tensorboard_path, params=cfg, metrics=metrics, sync_interval=cfg.train.transformer.eval_interval, remote_dir=remote_dir)

    print("######## Training ########")
    for epoch in range(epochs + 1):
        #
        # training epoch
        #
        model.train()
        for i, data in enumerate(tqdm.tqdm(train_dataloader)):
            # autoregressive loss transformer training
            optimizer.zero_grad()
            
            emb = data["embeddings"].to(device)
            
            vae_output = condition_model.forward(emb)
            timbre_cond = vae_output[1].detach()
            pitch_cond = vae_output[4].detach()

            # apply noise to condition vector for smoothing the output distribution
            # timbre_cond += torch.randn_like(timbre_cond).to(device) * torch.exp(0.5*vae_output[2])
            # pitch_cond += torch.randn_like(pitch_cond).to(device) * 0.05

            # concatenating timbre and pitch condition for putting into encoder of transformer
            combined_cond = torch.cat((timbre_cond, pitch_cond), dim=1)

            # emb_shifted is the transposed input vector (in time dimension) for autoregressive training
            emb_shifted = emb[:,:-1,:]
            emb_shifted = torch.cat((torch.zeros((emb_shifted.shape[0],1,128)).to(device),emb_shifted),dim=1).detach()
            
            logits = model.forward(xdec=emb_shifted, xenc=combined_cond)
            loss = loss_fn(logits, emb)
            
            loss.backward()
            optimizer.step()
            
        # adapt learning rate
        if epoch in map(lambda x: int(epochs*x), lr_change_intervals):
            # adapt learning rate with multiplier
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_change_multiplier
            print('changed learning rate to %.3e after epoch %d' % (optimizer.param_groups[0]['lr'], epoch))

        model.eval()
        if (epoch % cfg.train.transformer.eval_interval) == 0 and epoch > 0:
            print()
            losses = eval_model(model, condition_model, train_dataloader, device=device, num_batches=cfg.train.transformer.num_batches_evaluation, loss_fn=loss_fn)
            print("TRAIN: Epoch %d: Train loss: %.6f, Generation Loss: %.6f" % (epoch, losses[0], losses[1]))
            if writer is not None:
                writer.add_scalar("train/loss", losses[0], epoch)
                writer.add_scalar("train/gen_loss", losses[1], epoch)

            losses = eval_model(model, condition_model, valid_dataloader, device=device, num_batches=cfg.train.transformer.num_batches_evaluation, loss_fn=loss_fn)
            print("VALID: Epoch %d: Val loss: %.6f, Generation Loss: %.6f" % (epoch, losses[0], losses[1]))
            if writer is not None:
                writer.add_scalar("valid/loss", losses[0], epoch)
                writer.add_scalar("valid/gen_loss", losses[1], epoch)

        if (epoch % cfg.train.transformer.visualize_interval) == 0 and epoch > 0:
            print("Visualizing the model")
            visu_model(model, condition_model, train_dataloader, device=device, name_prefix="train", epoch=epoch, writer=writer)
            visu_model(model, condition_model, valid_dataloader, device=device, name_prefix="valid", epoch=epoch, writer=writer)

        if cfg.train.transformer.hear_interval > 0:
            if (epoch % cfg.train.transformer.hear_interval) == 0 and epoch > 0:
                print("Hearing the model")
                hear_model(model, condition_model, encodec_model, train_dataloader, device=device, num_examples=2, name_prefix="train", epoch=epoch, writer=writer)
                hear_model(model, condition_model, encodec_model, valid_dataloader, device=device, num_examples=2, name_prefix="valid", epoch=epoch, writer=writer)

        if (epoch % cfg.train.transformer.save_interval) == 0 and epoch > 0:
            print("Saving model at epoch %d" % (epoch))
            torch.save(model, 'out/transformer/checkpoints/transformer_epoch_%d.torch' % (epoch))

        if writer is not None and epoch > 0:
            writer.step()
            
    print("Training completed. Saving the model.")
    torch.save(model, 'out/transformer/checkpoints/transformer_final_epoch_%d.torch' % (epochs))
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()