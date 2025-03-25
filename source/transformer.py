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
        self.input_projection_encoder = nn.Linear(1,self.config['internal_dimension']).to(device)

        self.output_projection = nn.Linear(self.config['internal_dimension'], self.config['input_dimension']).to(device)

        # positional encoding layer decoder
        self.input_posemb_decoder = nn.Embedding(self.config['block_size'], self.config['internal_dimension']).to(device)
        # positional enocding layer encoder
        self.input_posemb_encoder = nn.Embedding(self.config['block_size_encoder'], self.config['internal_dimension']).to(device)
        
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
        xdec[:,0,:64] = xenc[:,0].unsqueeze(1).repeat(1,64)
        xdec[:,0,64:] = xenc[:,1].unsqueeze(1).repeat(1,64)
  
        xdec = self.input_projection_decoder(xdec)
        pos = torch.arange(0, xdec.shape[1], dtype=torch.long).to(self.device)
        pos_emb_dec = self.input_posemb_decoder(pos)
        xdec = xdec + pos_emb_dec
        
        xenc = self.input_projection_encoder(xenc.unsqueeze(-1))
        pos = torch.arange(0, self.config['block_size_encoder'], dtype=torch.long).to(self.device)
        pos_emb_enc = self.input_posemb_encoder(pos)
        xenc = xenc + pos_emb_enc
        
        mask = self.get_tgt_mask(xdec.shape[1])
        ydec = self.transformer.forward(src=xenc,tgt=xdec,tgt_mask=mask)
        ydec = self.output_projection(ydec)
        
        return ydec
    
    @torch.no_grad()
    def generate(self, num_generate, condition):
        """
        num_generate: output size of generated sequence (time dimension)
        condition: nx2 list of tensor (n=batch size)
        """
        if not torch.is_tensor(condition):
            condition = torch.tensor(condition, dtype=torch.float32).to(device=self.device)        
        gx = torch.zeros((condition.shape[0],1,self.config['input_dimension']), dtype=torch.float32).to(self.device) # all zeros is the 'start token'
            
        for _ in range(num_generate-1):
            ng = self.forward(xdec=gx,xenc=condition)[:, [-1], :]
            gx = torch.cat((gx, ng), dim=1)
        return gx
    

    def get_tgt_mask(self, size) -> torch.tensor:
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        return mask

