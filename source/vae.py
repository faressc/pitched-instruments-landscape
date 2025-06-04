
import torch
import torch.nn as nn
from torch.nn import functional as F

import math

class ConditionConvVAE(nn.Module):
    def __init__(self, channels, linears, input_crop, device, kernel_size=5, dilation=1, padding=None, stride=2, output_padding=1, dropout_ratio=0.2, num_notes=128, num_instruments=1006, num_families=11):
        super(ConditionConvVAE, self).__init__()
        if padding is None:
            padding = kernel_size//2
                 
                 
        # calculate all dimensions here   
        conv_sizes = [input_crop]
        for _ in range(len(channels)-1):
            cs = ( (conv_sizes[-1]+2*padding-dilation*(kernel_size-1)-1)/stride ) + 1
            conv_sizes.append(math.floor(cs))
            print(cs)
        intermediate_output_size = conv_sizes[-1] * channels[-1]
        deconv_sizes = [conv_sizes[-1]]
        for _ in range(len(channels)-1):
            dcs = (deconv_sizes[-1]-1) * stride - 2*padding + dilation * (kernel_size-1) + output_padding + 1
            deconv_sizes.append(dcs)
        
        
        
        linears = [intermediate_output_size,] + linears
        
        

        # encoder
        for i in range(len(channels)-1):
            setattr(self, 'enc_conv{}'.format(i), nn.Conv1d(
                channels[i], channels[i+1], kernel_size=kernel_size, stride=stride, padding=padding
            ))
            setattr(self, 'enc_conv_norm{}'.format(i), nn.LayerNorm([channels[i+1], conv_sizes[i+1]]))
        
        for i in range(len(linears)-2):
            setattr(self, 'enc_lin{}'.format(i), nn.Linear(
                linears[i], linears[i+1],
            ))
            setattr(self, 'enc_lin_norm{}'.format(i), nn.LayerNorm(linears[i+1]))


        self.head_size = linears[-2]//2


        self.note_cls_head = torch.nn.Linear(linears[-2], self.head_size)
        self.regression_head = torch.nn.Linear(linears[-2], self.head_size)

        self.note_cls_head_norm = nn.LayerNorm(self.head_size)
        self.regression_head_norm = nn.LayerNorm(self.head_size)

        # fed by regression head
        self.mean = nn.Linear(self.head_size, linears[-1])
        self.logvar = nn.Linear(self.head_size, linears[-1])
        
        # fed by classification head
        self.note_cls = nn.Linear(self.head_size, num_notes)
        
        self.family_cls_head_linears = [linears[-1], 4, 8, 16, 32, num_families]

        self.instrument_cls_head_linears = [linears[-1], 4, 8, 16, 32, 64, 32, 64, 128, 256, num_instruments]

        for hn in range(len(self.family_cls_head_linears)-1):
            l1 = self.family_cls_head_linears[hn]
            l2 = self.family_cls_head_linears[hn+1]
            setattr(self, 'family_cls_head_{}'.format(hn), nn.Linear(l1,l2))
            setattr(self, 'family_cls_head_norm_{}'.format(hn), nn.LayerNorm(l2))

        for hn in range(len(self.instrument_cls_head_linears)-1):
            l1 = self.instrument_cls_head_linears[hn]
            l2 = self.instrument_cls_head_linears[hn+1]
            setattr(self, 'instrument_cls_head_{}'.format(hn), nn.Linear(l1,l2))
            setattr(self, 'instrument_cls_head_norm_{}'.format(hn), nn.LayerNorm(l2))
        
        dec_linears = linears[::-1]
        dec_channels = channels[::-1]

        dec_linears[0] += num_notes

        # decoder
        # Fully connected layers
        for i in range(len(dec_linears)-1):
            setattr(self, 'dec_lin{}'.format(i), nn.Linear(dec_linears[i], dec_linears[i+1]))
            setattr(self, 'dec_lin_norm{}'.format(i), nn.LayerNorm(dec_linears[i+1]))
            
        # make deconvolutional layers
        for i in range(len(dec_channels)-1):
            setattr(self, 'dec_deconv{}'.format(i), nn.ConvTranspose1d(
                dec_channels[i], dec_channels[i+1], kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding
            ))
            setattr(self, 'dec_deconv_norm{}'.format(i), nn.LayerNorm([dec_channels[i+1], deconv_sizes[i+1]]))
    
        self.relu = torch.nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout_ratio)

        self.linears = linears
        self.channels = channels
        self.dec_linears = dec_linears
        self.dec_channels = dec_channels
        self.conv_sizes = conv_sizes
        self.deconv_sizes = deconv_sizes
        
        print('conv sizes', conv_sizes)
        print('linears', linears)
        print('dec linears', dec_linears)
        print('dec deconvs', deconv_sizes)
        
        self.initialize_weights()

        self.channels = channels
        self.linears = linears
        self.input_crop = input_crop
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        self.output_padding = output_padding
        self.device = device
        self.num_notes = num_notes
        self = self.to(device)

    def forward(self, input, encoder_only=False):
        input = input.to(self.device)
        mean, logvar, note_cls, family_cls, instrument_cls = self.encode(input)  # Assume logvar is actually logvar

        epsilon = torch.randn_like(logvar).to(self.device)  # Sampling epsilon
        std = torch.exp(0.5 * logvar)  # Convert log variance to standard deviation
        x = mean + std * epsilon  # Reparameterization trick

        max_indices = torch.argmax(note_cls, dim=-1)

        # Create a one-hot tensor
        one_hot = torch.nn.functional.one_hot(max_indices, num_classes=note_cls.shape[-1]).float()
        # Detach to prevent gradient flow
        one_hot = one_hot.detach()
        

        # Reparameterization trick
        if encoder_only:
            return mean, logvar, note_cls, one_hot, family_cls, instrument_cls
        
        x = torch.cat((x,one_hot), dim=1)

        x = self.decode(x)
        
        return x, mean, logvar, note_cls, one_hot, family_cls, instrument_cls


    def encode(self, x):
        x = x.to(self.device)
        x = x[:,:self.input_crop,:]
        x = x.swapaxes(1,2) # for making it batch, channels, time
        
        # encoder
        for i in range(0,len(self.channels)-1):
            x = getattr(self, 'enc_conv{}'.format(i))(x)
            x = getattr(self, 'enc_conv_norm{}'.format(i))(x)
            x = self.relu(x)
            x = self.dropout(x)
        
        x = x.view([x.shape[0], -1] )
        for i in range(0,len(self.linears)-2):
            x = getattr(self, 'enc_lin{}'.format(i))(x)
            x = getattr(self, 'enc_lin_norm{}'.format(i))(x)
            x = self.relu(x)
            x = self.dropout(x)

        x_reg = self.regression_head(x)
        x_reg = self.regression_head_norm(x_reg)
        x_reg = self.relu(x_reg)

        x_cla = self.note_cls_head(x)
        x_cla = self.note_cls_head_norm(x_cla)
        x_cla = self.relu(x_cla)

        mean = self.mean(x_reg)
        logvar = self.logvar(x_reg)
        
        note_cls = self.note_cls(x_cla)
        
        family_cls = mean.clone()
        # family_cls += torch.randn(size=family_cls.shape, device=self.device) * 0.05
        instrument_cls = mean.clone()

        for i in range(len(self.family_cls_head_linears)-2):
            family_cls = getattr(self, 'family_cls_head_{}'.format(i))(family_cls)
            family_cls = getattr(self, 'family_cls_head_norm_{}'.format(i))(family_cls)
            family_cls = self.relu(family_cls)
            if i > 3:
                family_cls = self.dropout(family_cls)

        family_cls = getattr(self, 'family_cls_head_{}'.format(len(self.family_cls_head_linears)-2))(family_cls)
            
        for i in range(len(self.instrument_cls_head_linears)-2):
            instrument_cls = getattr(self, 'instrument_cls_head_{}'.format(i))(instrument_cls)
            instrument_cls = getattr(self, 'instrument_cls_head_norm_{}'.format(i))(instrument_cls)
            instrument_cls = self.relu(instrument_cls)
            if i > 3:
                instrument_cls = self.dropout(instrument_cls)

        instrument_cls = getattr(self, 'instrument_cls_head_{}'.format(len(self.instrument_cls_head_linears)-2))(instrument_cls)

        return mean, logvar, note_cls, family_cls, instrument_cls

    def decode(self, x):
        x = x.to(self.device)
        # decoder
        for i in range(0,len(self.linears)-1):
            x = getattr(self, 'dec_lin{}'.format(i))(x)
            x = getattr(self, 'dec_lin_norm{}'.format(i))(x)
            x = self.relu(x)
            if i > 1:
                x = self.dropout(x)

        x = x.view([x.shape[0], self.dec_channels[0], self.deconv_sizes[0]]) # batch, channels, time

        for i in range(0,len(self.channels)-1):
            x = getattr(self, 'dec_deconv{}'.format(i))(x)
            x = getattr(self, 'dec_deconv_norm{}'.format(i))(x)
            if i < len(self.channels)-2:
                x = self.relu(x)
                x = self.dropout(x)

        x = x[:,:,:self.input_crop] # crop it to input crop for calculating loss etc
        x = x.swapaxes(1,2) # for making it batch, time, channels
        
        return x

    def initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv1d):
                nn.init.xavier_normal_(layer.weight, gain=2)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            if isinstance(layer, nn.ConvTranspose1d):
                nn.init.xavier_normal_(layer.weight, gain=2)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)  # Initialize biases to zero or another suitable value
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=2)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)  # Initialize biases to zero or another suitable value


def calculate_vae_loss(x, x_hat, mean, logvar, note_cls, gt_note_cls, family_cls, gt_family, instrument_cls, gt_inst, current_epoch, num_epochs, weighted_reproduction, loss_fn, cls_loss_fn, batch_size, device, rec_beta, neighbor_beta, spa_beta, kl_beta, note_cls_beta, family_cls_beta, instrument_cls_beta, reg_scaling_exp_neighbor, reg_scaling_exp_family, reg_scaling_exp_instrument, note_remap, instrument_remap):

    b, t, f = x.shape
    
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
    
    note_cls_loss = cls_loss_fn(note_cls, gt_note_cls)
    family_cls_loss = cls_loss_fn(family_cls, gt_family)
    instrument_cls_loss = cls_loss_fn(instrument_cls, gt_inst)

    #
    # reproduction loss
    #
    reproduction_loss = None
    rep_norm = b*t*f
    if weighted_reproduction:
        # weighting function for increasing the weight of the beginning of the sample
        # idea would be envolope loss (adsr)
        weight = torch.zeros_like(x)
        for i in range(x.shape[1]):
            weight[:,i,:] = 1-(i/x.shape[1])**2
        weight = 1/weight.mean() * weight
        reproduction_loss = loss_fn(x_hat*weight, x*weight) / rep_norm
    else:
        reproduction_loss = loss_fn(x_hat, x) / rep_norm
    
    # neighboring loss
    dists = torch.cdist(mean, mean, p=2)  # shape: [B, B]
    
    # Create label comparison matrix
    gt_inst_i = gt_inst.unsqueeze(1).expand(-1, b)
    gt_inst_j = gt_inst.unsqueeze(0).expand(b, -1)
    
    same_class = (gt_inst_i == gt_inst_j).float()
    diff_class = 1.0 - same_class

    eps = 1e-6
    margin = 0.25  # margin for pushing away different classes

    # Attractive loss (same class): encourage close latent vectors
    attractive_loss = (same_class * dists**2).sum() / (same_class.sum() + eps)

    # Repulsive loss (different class): encourage margin between different classes
    repulsive_loss = (diff_class * torch.clamp(margin - dists, min=0)**2).sum() / (diff_class.sum() + eps)

    neighbor_loss = attractive_loss + repulsive_loss

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / batch_size

    # # spatial regularization loss
    l_orig = torch.linalg.vector_norm(mean,ord=2,dim=1)
    zero_tensor = torch.FloatTensor([0.0]).to(device)
    spatial_loss = torch.max((l_orig-1.0),zero_tensor.expand_as(l_orig)).mean()

    training_progress = (current_epoch/num_epochs)

    def regularization_scaling(x, reg_scaling_exp):
        return x**reg_scaling_exp
    
    return rec_beta * reproduction_loss, \
    neighbor_beta * regularization_scaling(training_progress, reg_scaling_exp_neighbor) * neighbor_loss, \
    spa_beta * spatial_loss, \
    kl_beta * kl_loss, \
    note_cls_beta * note_cls_loss, \
    (1-regularization_scaling(training_progress, reg_scaling_exp_family)) * family_cls_beta * family_cls_loss, \
    regularization_scaling(training_progress, reg_scaling_exp_instrument) * instrument_cls_beta * instrument_cls_loss