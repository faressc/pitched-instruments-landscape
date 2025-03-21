import utils.debug

import os

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

import utils.ffmpeg_helper as ffmpeg

import numpy as np
import matplotlib.pyplot as plt

def eval_model(model, dl, device, loss_fn, input_crop):
    losses = []
    cls_pred = []
    cls_gt = []
    for i, data in enumerate(dl):
        emb = data['embeddings'].to(device)
        emb = emb.view(-1,128,300).permute(0,2,1)
        emb = normalize_embedding(emb)
        emb_pred, mean, var, note_cls = model.forward(emb)
        gt_cls = data['metadata']['pitch'].to(device)
        rec_loss, reg_loss, cls_loss = loss_fn(emb[:,:input_crop,:], emb_pred, mean, var, note_cls, gt_cls, loss_fn.keywords['epochs'])
        losses.append([rec_loss.item(), reg_loss.item(), cls_loss.item()])
        
        cls_pred.extend(note_cls.argmax(dim=1).cpu().numpy())
        cls_gt.extend(gt_cls.cpu().numpy())
        
        if i > 50: # data set can be very large
            break
        
    b = np.array(cls_pred) == np.array(cls_gt)    
    acc01 = np.count_nonzero(b) / len(b)
    
    rec_loss, reg_loss, cls_loss = np.array(losses).mean(axis=0)
    return rec_loss, reg_loss, cls_loss, acc01

def visu_model(model, dl, device, input_crop, name_prefix=''):
    num_samples = 500
    
    embs = np.zeros((0,input_crop,128))
    embs_pred = np.zeros((0,input_crop,128))
    means = np.zeros((0,2))
    families = np.zeros((0))
    for i, data in enumerate(dl):
        emb = data['embeddings'].to(device)
        emb = emb.view(-1,128,300).permute(0,2,1)
        emb = normalize_embedding(emb)
        emb_pred, mean, var, note_cls = model.forward(emb)
        embs = np.vstack((embs,emb[:,:input_crop,:].cpu().detach().numpy()))
        embs_pred = np.vstack((embs_pred,emb_pred.cpu().detach().numpy()))
        means = np.vstack((means, mean.cpu().detach().numpy()))
        families = np.concat((families, data['metadata']['family'].numpy()))
        if len(embs) > num_samples: # skip when ds gets too large
            break
    
    
    # plot original embedding and decoded embedding
    save_index = 1
    fig1, axes1 = plt.subplots(1, 2, figsize=(10, 5), num=1)  # 1 row, 2 columns
    # orig embedding
    axes1[0].imshow(embs[save_index], vmin=0.0, vmax=1.0)
    axes1[0].set_title("Original")
    # generated embedding
    axes1[1].imshow(embs_pred[save_index], vmin=0.0, vmax=1.0)
    axes1[1].set_title("Generated")
    plt.savefig('out/%s_embedding_comparison.png' % (name_prefix,))
    fig1.clear()
    plt.close(1)  # Schließt die bestehende Figur mit Nummer 1

    # plot the latent as scatters
    fig2 = plt.figure(2)
    plt.scatter(means[:,0], means[:,1], c=families)
    plt.savefig('out/%s_latent_visualization.png' % (name_prefix,))
    fig2.clear()
    plt.close(2) 



def normalize_embedding(emb):
    return (emb + 25.) / (33.5 + 25.)

def denormalize_embedding(normalized_embedding):
    return normalized_embedding * (33.5 + 25.) - 25.


def main():
    print("##### Starting Train Stage #####")
    os.makedirs("out/checkpoints", exist_ok=True)

    eval_epoch = 2
    visu_epoch = 2

    # Load the parameters from the dictionary into variables
    cfg = OmegaConf.load("params.yaml")
    

    # Set a random seed for reproducibility across all devices. Add more devices if needed
    config.set_random_seeds(cfg.train.random_seed)
    # Prepare the requested device for training. Use cpu if the requested device is not available 
    device = config.auto_device()

    print(f"Creating the valid dataset and dataloader with db_path: {cfg.train.db_path_valid}")
    train_dataset = MetaAudioDataset(db_path=cfg.train.db_path_train)
    # train_dataset = MetaAudioDataset(db_path=cfg.train.db_path_train, max_num_samples=1000) # for testing
    # train_dataset = MetaAudioDataset(db_path=cfg.train.db_path_train, max_num_samples=10000, has_audio=False) # no audio data in the dataset
    valid_dataset = MetaAudioDataset(db_path=cfg.train.db_path_valid)
    # filter_pitch_sampler = FilterPitchSampler(dataset=valid_dataset, pitch=cfg.train.pitch)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.train.vae.batch_size,
                                  # sampler=FilterPitchSampler(valid_dataset, cfg.train.pitch, True),
                                  drop_last=False,
                                  num_workers=cfg.train.num_workers,
                                  shuffle=True,
                                  )

    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=cfg.train.vae.batch_size,
                                  # sampler=FilterPitchSampler(valid_dataset, cfg.train.pitch, False),
                                  drop_last=False,
                                  num_workers=cfg.train.num_workers,
                                  shuffle=False,
                                )
    

    
    vae = ConditionConvVAE(cfg.train.vae.channels, cfg.train.vae.linears, cfg.train.vae.input_crop, device=device, dropout_ratio=cfg.train.vae.dropout_ratio, num_notes=128)
    encodec_model = EncodecModel.from_pretrained(cfg.preprocess.model_name).to(device)

    optimizer = torch.optim.AdamW(vae.parameters(), lr=cfg.train.vae.lr, weight_decay=cfg.train.vae.wd, betas=cfg.train.vae.betas)
    calculate_vae_loss = instantiate(cfg.train.vae.calculate_vae_loss, _recursive_=True)
    calculate_vae_loss = partial(calculate_vae_loss, device=device)

    for epoch in range(cfg.train.vae.epochs):
        #
        # training epoch
        # 
        vae.train()
        for i, data in enumerate(train_dataloader):
            # print(f"Data audio_data[{i}]: {data['audio_data'].shape}")
            # print(f"Data metadata[{i}]: {data['metadata']['pitch'].shape}")
            # print(f"Data embeddings[{i}]: {data['embeddings'].shape}")
            optimizer.zero_grad()
            
            emb = data['embeddings'].view(-1,128,300).permute(0,2,1)
            emb = normalize_embedding(emb)
            emb = emb.to(device)
            emb_pred, mean, var, note_cls = vae.forward(emb)
            
            rec_loss, reg_loss, cls_loss = calculate_vae_loss(emb[:,:cfg.train.vae.input_crop,:], emb_pred, mean, var, note_cls, data['metadata']['pitch'].to(device), epoch)
            
            loss = rec_loss + reg_loss + cls_loss
            loss.backward()
            optimizer.step()
                    
        if epoch == int(0.6*cfg.train.vae.epochs) or epoch == int(0.8*cfg.train.vae.epochs) or epoch == int(0.9*cfg.train.vae.epochs):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.3
            print('decreased learning rate to %.8f' % (param_group['lr'],))
        
        #
        # evaluate model
        # 
        vae.eval()
        if (epoch % eval_epoch) == 0 and epoch > 0:
            # reconstruction error on validation dataset
            print()
            losses = eval_model(vae, train_dataloader, device, calculate_vae_loss, cfg.train.vae.input_crop)
            print("TRAIN: Epoch %d: Reconstruction loss: %.6f, Regularization Loss: %.6f, Classifier Loss: %.6f, Classifier 0/1 Accuracy: %.6f" % (epoch, losses[0], losses[1], losses[2], losses[3]))
            losses = eval_model(vae, valid_dataloader, device, calculate_vae_loss, cfg.train.vae.input_crop)
            print("VAL: Epoch %d: Reconstruction loss: %.6f, Regularization Loss: %.6f, Classifier Loss: %.6f, Classifier 0/1 Accuracy: %.6f" % (epoch, losses[0], losses[1], losses[2], losses[3]))
        
            torch.save(vae, 'out/checkpoints/vae.torch')
        if (epoch % visu_epoch) == 0 and epoch > 0:
            # visual evaluation on validation dataset
            visu_model(vae, train_dataloader, device, cfg.train.vae.input_crop, name_prefix='train')
            visu_model(vae, valid_dataloader, device, cfg.train.vae.input_crop, name_prefix='val')
            # save audio
            num_generate = 5
            emb_pred_for_audio = denormalize_embedding(emb_pred[:num_generate])
            decoded = encodec_model.decoder((emb_pred_for_audio).permute(0,2,1))
            decoded = decoded.detach().cpu().numpy()
            decoded_int = np.int16(decoded * (2**15 - 1))
            for ind in range(num_generate):
                decoded_sample = decoded_int[ind]
                ffmpeg.write_audio_file(decoded_sample, "out/vae_generated_%d.wav" % (ind,), 24000)
            
            
            
    print("Gone through the dataset")
    # # Create a SummaryWriter object to write the tensorboard logs
    # tensorboard_path = logs.return_tensorboard_path()
    # metrics = {'Epoch_Loss/train': None, 'Epoch_Loss/test': None, 'Batch_Loss/train': None}
    # writer = logs.CustomSummaryWriter(log_dir=tensorboard_path, params=params, metrics=metrics)

    # # Set a random seed for reproducibility across all devices. Add more devices if needed
    # config.set_random_seeds(random_seed)
    # # Prepare the requested device for training. Use cpu if the requested device is not available 
    # device = config.prepare_device(device_request)

    # # Load preprocessed data from the input file into the training and testing tensors
    # input_file_path = Path('data/processed/data.pt')
    # data = torch.load(input_file_path)
    # X_ordered_training = data['X_ordered_training']
    # y_ordered_training = data['y_ordered_training']
    # X_ordered_testing = data['X_ordered_testing']
    # y_ordered_testing = data['y_ordered_testing']

    # # Create the model
    # model = NeuralNetwork(conv1d_filters, conv1d_strides, hidden_units).to(device)
    # summary = torchinfo.summary(model, (1, 1, input_size), device=device)
    # print(summary)

    # # Add the model graph to the tensorboard logs
    # sample_inputs = torch.randn(1, 1, input_size) 
    # writer.add_graph(model, sample_inputs.to(device))

    # # Define the loss function and the optimizer
    # loss_fn = torch.nn.MSELoss(reduction='mean')
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # # Create the dataloaders
    # training_dataset = torch.utils.data.TensorDataset(X_ordered_training, y_ordered_training)
    # training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    # testing_dataset = torch.utils.data.TensorDataset(X_ordered_testing, y_ordered_testing)
    # testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)

    # # Training loop
    # for t in range(epochs):
    #     print(f"Epoch {t+1}\n-------------------------------")
    #     epoch_loss_train = train_epoch(training_dataloader, model, loss_fn, optimizer, device, writer, epoch=t)
    #     epoch_loss_test = test_epoch(testing_dataloader, model, loss_fn, device, writer)
    #     epoch_audio_prediction, epoch_audio_target  = generate_audio_examples(model, device, testing_dataloader)
    #     writer.add_scalar("Epoch_Loss/train", epoch_loss_train, t)
    #     writer.add_scalar("Epoch_Loss/test", epoch_loss_test, t)
    #     writer.add_audio("Audio/prediction", epoch_audio_prediction, t, sample_rate=44100)
    #     writer.add_audio("Audio/target", epoch_audio_target, t, sample_rate=44100)        
    #     writer.step()  

    # writer.close()

    # # Save the model checkpoint
    # output_file_path = Path('models/checkpoints/model.pth')
    # output_file_path.parent.mkdir(parents=True, exist_ok=True)
    # torch.save(model.state_dict(), output_file_path)
    # print("Saved PyTorch Model State to model.pth")
    

if __name__ == "__main__":
    main()
