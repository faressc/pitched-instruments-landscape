
import torch
import torch.nn as nn
from torch.nn import functional as F

import math



class ConditionConvVAE(nn.Module):
    def __init__(self, channels, linears, input_crop, device, kernel_size=5, dilation=1, padding=None, stride=2, output_padding=1, dropout_ratio=0.2, num_notes=128):
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


        self.classification_head = torch.nn.Linear(linears[-2], self.head_size)
        self.regression_head = torch.nn.Linear(linears[-2], self.head_size)

        self.cla_head_norm = nn.LayerNorm(self.head_size)
        self.reg_head_norm = nn.LayerNorm(self.head_size)

        # fed by regression head
        self.mean = nn.Linear(self.head_size, linears[-1])
        self.var = nn.Linear(self.head_size, linears[-1])
        
        # fed by classification head
        self.note_cls = nn.Linear(self.head_size, num_notes)
        
        self.instrument_head_linears = [linears[-1], 4, 8, 16, 32, 64, 32, 64, 128, 256, 512, 1024]
        
        for hn in range(len(self.instrument_head_linears)-1):
            l1 = self.instrument_head_linears[hn]
            l2 = self.instrument_head_linears[hn+1]
            setattr(self, 'instrument_head_{}'.format(hn), nn.Linear(l1,l2))
            setattr(self, 'instrument_head_norm_{}'.format(hn), nn.LayerNorm(l2))
        

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
        mean, logvar, note_cls, cls_head = self.encode(input)  # Assume var is actually logvar

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
            return mean, logvar, note_cls, one_hot, cls_head
        
        x = torch.cat((x,one_hot), dim=1)

        x = self.decode(x)
        
        return x, mean, logvar, note_cls, one_hot, cls_head


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
        x_reg = self.reg_head_norm(x_reg)
        x_reg = self.relu(x_reg)

        x_cla = self.classification_head(x)
        x_cla = self.cla_head_norm(x_cla)
        x_cla = self.relu(x_cla)

        mean = self.mean(x_reg)
        var = self.var(x_reg)
        
        note_cls = self.note_cls(x_cla)
        
        
        cls_head = mean.clone()
        
        cls_head += torch.randn(size=cls_head.shape, device=self.device) * 0.05
        
        for i in range(len(self.instrument_head_linears)-2):
            cls_head = getattr(self, 'instrument_head_{}'.format(i))(cls_head)
            cls_head = getattr(self, 'instrument_head_norm_{}'.format(i))(cls_head)
            cls_head = self.relu(cls_head)
            if i > 3:
                cls_head = self.dropout(cls_head)

        cls_head = getattr(self, 'instrument_head_{}'.format(len(self.instrument_head_linears)-2))(cls_head)
            
        return mean, var, note_cls, cls_head

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
                nn.init.xavier_normal_(layer.weight, gain=1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            if isinstance(layer, nn.ConvTranspose1d):
                nn.init.xavier_normal_(layer.weight, gain=1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)  # Initialize biases to zero or another suitable value
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)  # Initialize biases to zero or another suitable value


# TODO: Add KL divergence loss
def calculate_vae_loss(x, x_hat, mean, var, note_cls, gt_cls, cls_head, gt_inst, current_epoch, num_epochs, weighted_reproduction, loss_fn, cls_loss_fn, batch_size, device, rec_beta, rep_beta, spa_beta, cla_beta, inst_beta, reg_scaling_exp):
    
    b, t, f = x.shape
    
    
    cls_loss = cls_loss_fn(note_cls, gt_cls)
    
    instr_loss = cls_loss_fn(cls_head, gt_inst)
    
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

    #
    # regularization loss terms
    #
    
    
    
    
    
    
    # neighboring loss
    dists = torch.cdist(mean, mean, p=2)  # shape: [B, B]
    
    # Create label comparison matrix
    gt_inst_i = gt_inst.unsqueeze(1).expand(-1, b)
    gt_inst_j = gt_inst.unsqueeze(0).expand(b, -1)
    
    same_class = (gt_inst_i == gt_inst_j).float()
    diff_class = 1.0 - same_class

    eps = 1e-6
    margin = 1.0  # margin for pushing away different classes

    # Attractive loss (same class): encourage close latent vectors
    attractive_loss = (same_class * dists**2).sum() / (same_class.sum() + eps)

    # Repulsive loss (different class): encourage margin between different classes
    repulsive_loss = (diff_class * torch.clamp(margin - dists, min=0)**2).sum() / (diff_class.sum() + eps) * 0.1

    neighbor_loss = attractive_loss + repulsive_loss


    kl_loss = 0.1 * -0.5 * torch.sum(1 + var - mean.pow(2) - var.exp()) / batch_size


    # # spatial regularization loss
    l_orig = torch.linalg.vector_norm(mean,ord=2,dim=1)
    zero_tensor = torch.FloatTensor([0.0]).to(device)
    spatial_loss = torch.max((l_orig-1.0),zero_tensor.expand_as(l_orig)).mean()
    
    # latent_norms = torch.norm(mean, dim=1)  # shape: [B]
    # spatial_loss = torch.mean((latent_norms - 1.0)**2)

    
    
    # # old linear way
    warmup_ratio_rep = 0.0
    warmup_ratio_inst = 0.0

    
    training_progress = (current_epoch/num_epochs)
    warmup_beta_rep = 0.0 if training_progress < warmup_ratio_rep else training_progress
    warmup_beta_inst_cls = 0.0 if training_progress < warmup_ratio_inst else training_progress



    def regularization_scaling(x, reg_scaling_exp):
        return x**reg_scaling_exp
    

    return rec_beta * reproduction_loss, \
    rep_beta * regularization_scaling(warmup_beta_rep, reg_scaling_exp) * neighbor_loss, \
    spa_beta * spatial_loss + kl_loss, \
    cla_beta * cls_loss, \
    warmup_beta_inst_cls * inst_beta * instr_loss