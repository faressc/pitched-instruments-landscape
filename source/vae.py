
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
        
        self.cls_head0 = nn.Linear(linears[-1], 4)
        self.cls_head1 = nn.Linear(4, 8)
        self.cls_head2 = nn.Linear(8, 16)
        self.cls_head3 = nn.Linear(16, 32)
        self.cls_head_out = nn.Linear(32, 11)

        self.cls_head0_norm = nn.LayerNorm(4)
        self.cls_head1_norm = nn.LayerNorm(8)
        self.cls_head2_norm = nn.LayerNorm(16)
        self.cls_head3_norm = nn.LayerNorm(32)


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
        
        cls_head = self.cls_head0(mean)
        cls_head = self.cls_head0_norm(cls_head)
        cls_head = self.cls_head1(cls_head)
        cls_head = self.cls_head1_norm(cls_head)
        cls_head = self.cls_head2(cls_head)
        cls_head = self.cls_head2_norm(cls_head)
        cls_head = self.cls_head3(cls_head)
        cls_head = self.cls_head3_norm(cls_head)
        cls_head = self.cls_head_out(cls_head)
        
        return mean, var, note_cls, cls_head

    def decode(self, x):
        x = x.to(self.device)
        # decoder
        for i in range(0,len(self.linears)-1):
            x = getattr(self, 'dec_lin{}'.format(i))(x)
            x = getattr(self, 'dec_lin_norm{}'.format(i))(x)
            x = self.relu(x)
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
    min_dist = 2 / math.sqrt(batch_size) # distance between samples that is desired
    
    dists = torch.cdist(mean, mean, p=2) + torch.eye(mean.size(0), device=mean.device) * 1e-8
    
    eps = 1e-3
    dists = torch.where(dists < min_dist, dists, torch.ones_like(dists)*1000000)
    repulsion_effect = 1.0 / (dists + eps)
    
    mask = torch.eye(dists.size(0), device=device).bool()
    repulsion_effect = repulsion_effect.masked_fill_(mask, 0)
    neighbor_loss = repulsion_effect.sum() / (b * (1/eps))


    # spatial regularization loss
    l_orig = torch.linalg.vector_norm(mean,ord=2,dim=1)
    zero_tensor = torch.FloatTensor([0.0]).to(device)
    spatial_loss = torch.max((l_orig-1.0),zero_tensor.expand_as(l_orig)).mean()
    
    # # old linear way
    # warmup_ratio = 0.3
    # warmup_beta = 0.0 if training_progress < warmup_ratio else training_progress

    
    training_progress = (current_epoch/num_epochs)

    def regularization_scaling(x, reg_scaling_exp):
        return x**reg_scaling_exp

    return rec_beta * reproduction_loss, \
    rep_beta * regularization_scaling(training_progress, reg_scaling_exp) * neighbor_loss, \
    spa_beta * spatial_loss, \
    cla_beta * cls_loss, \
    inst_beta * instr_loss