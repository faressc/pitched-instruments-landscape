import random
import numpy as np
import gc
import matplotlib.pyplot as plt
import os

import torch
import torch.nn.functional as F
import torch.nn as nn


class GesamTransformer(nn.Module):
    def __init__(self, 
                 transformer_config,
                 device
                 ):
        super().__init__()
        self.config = transformer_config
        self.device = device
        
        self.input_projection_decoder = nn.Linear(self.config['input_dimension'], self.config['internal_dimension']).to(device)
        self.input_projection_encoder_timbre = nn.Linear(1,self.config['internal_dimension']).to(device)
        self.input_projection_encoder_pos = nn.Embedding(self.config['num_notes'],self.config['internal_dimension']).to(device)

        self.output_projection = nn.Linear(self.config['internal_dimension'], self.config['input_dimension']).to(device)

        # positional encoding layer decoder
        self.input_posemb_decoder = nn.Embedding(self.config['block_size'], self.config['internal_dimension']).to(device)
        # positional enocding layer encoder
        self.input_posemb_encoder = nn.Embedding(3, self.config['internal_dimension']).to(device)
        
        self.transformer = nn.Transformer(
                d_model = self.config['internal_dimension'], 
                batch_first=True, 
                nhead=self.config['n_head'],
                num_encoder_layers=self.config['n_layer_encoder'],
                num_decoder_layers=self.config['n_layer_decoder'],
                dropout=self.config['dropout'],
                dim_feedforward=self.config['feedforward_dimension']
        ).to(device)


    def forward(self, xdec, xenc):
        # TODO: Think about adding the pitch to the encoder input
        # xdec[:,0,:64] = xenc[:,0].unsqueeze(1).repeat(1,64)
        # xdec[:,0,64:] = xenc[:,1].unsqueeze(1).repeat(1,64)
  
        xdec = self.input_projection_decoder(xdec)
        pos = torch.arange(0, xdec.shape[1], dtype=torch.long).to(self.device)
        pos_emb_dec = self.input_posemb_decoder(pos)
        # TODO: Try out concatenating the positional embedding to the input
        xdec = xdec + pos_emb_dec
        
        xenc_timbre = xenc[:,:2]
        xenc_timbre = self.input_projection_encoder_timbre(xenc_timbre.unsqueeze(-1))
        xenc_pitch = xenc[:,2:]
        xenc_pitch = self.input_projection_encoder_pos(torch.argmax(xenc_pitch, dim=1)).unsqueeze(1)
        x_concat = torch.concat((xenc_timbre, xenc_pitch), dim=1)
        
        mask = self.get_tgt_mask(xdec.shape[1])
        ydec = self.transformer.forward(src=x_concat,tgt=xdec,tgt_mask=mask)
        ydec = self.output_projection(ydec)
        
        return ydec
    
    @torch.no_grad()
    def generate(self, num_tokens, condition):
        """
        num_tokens: output size of generated sequence (time dimension)
        condition: nx2 list of tensor (n=batch size)
        """
        if not torch.is_tensor(condition):
            condition = torch.tensor(condition, dtype=torch.float32).to(device=self.device)        
        start_token = torch.zeros((condition.shape[0],1,self.config['input_dimension']), dtype=torch.float32).to(self.device) # all zeros is the 'start token'
            
        for _ in range(num_tokens):
            next_token = self.forward(xdec=start_token,xenc=condition)[:, [-1], :]
            start_token = torch.cat((start_token, next_token), dim=1)

        return start_token[:, 1:, :] # remove the start token
    

    def get_tgt_mask(self, size) -> torch.tensor:
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        return mask
