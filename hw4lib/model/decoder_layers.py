import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, CrossAttentionLayer, FeedForwardLayer

'''
TODO: Implement these Modules.

This file contains two key decoder layer implementations used in transformer architectures:

1. SelfAttentionDecoderLayer: Used in decoder-only transformers (like GPT)
   - Contains masked self-attention and feed-forward sublayers
   - Used for tasks like language modeling where only previous tokens can be attended to
   
2. CrossAttentionDecoderLayer: Used in encoder-decoder transformers (like BART)
   - Contains masked self-attention, cross-attention, and feed-forward sublayers
   - Used for tasks like translation where decoder needs to attend to encoder outputs

Each layer follows a Pre-LN (Layer Normalization) architecture where:
- Layer normalization is applied before each sublayer operation
- Residual connections wrap around each sublayer

Implementation Steps for Each Layer:
1. Initialize the required sublayers in __init__:
   - SelfAttentionLayer for masked self-attention
   - CrossAttentionLayer for cross-attention (in CrossAttentionDecoderLayer only)
   - FeedForwardLayer for position-wise processing

2. Implement the forward pass to:
   - Apply sublayers in the correct order
   - Pass appropriate masks to attention layers
   - Return both outputs and attention weights
'''

## -------------------------------------------------------------------------------------------------  
## Decoder Layers
## -------------------------------------------------------------------------------------------------      
class SelfAttentionDecoderLayer(nn.Module):
    '''
    Pre-LN Decoder Layer with masked self-attention and feed-forward sublayers.
    Used in the decoder-only Transformer architecture.  
    '''
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        '''
        Initialize the SelfAttentionDecoderLayer. 
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            d_ff      (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
        ''' 
        super().__init__()
        # TODO: Implement __init__
       
        # TODO: Initialize the sublayers      
        self.self_attn = SelfAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        ) # Masked self-attention layer
        self.ffn = FeedForwardLayer(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        ) # Feed-forward network

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the DecoderLayer1.
        Args:
            x (torch.Tensor): The input tensor. shape: (batch_size, seq_len, d_model)   
            key_padding_mask (torch.Tensor): The padding mask for the decoder. shape: (batch_size, seq_len)
            attn_mask (torch.Tensor): The self-attention mask. shape: (seq_len, seq_len)

        Returns:
            x (torch.Tensor): The output tensor. shape: (batch_size, seq_len, num_classes)
            mha_attn_weights (torch.Tensor): The attention weights. shape: (batch_size, seq_len, seq_len)   
        '''
        # TODO: Implement forward: Follow the figure in the writeup

        x, mha_attn_weights = self.self_attn(
            x=x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        output = self.ffn(x)
        # TODO: Return the output tensor and attention weights
        return output, mha_attn_weights

## -------------------------------------------------------------------------------------------------    
class CrossAttentionDecoderLayer(nn.Module):
    '''
    Pre-LN Decoder Layer with masked self-attention, cross-attention, and feed-forward sublayers.
    Used in the encoder-decoder Transformer architecture.
    '''
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        '''
        Initialize the CrossAttentionDecoderLayer. 
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            d_ff      (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
        '''
        super().__init__()
        # TODO: Implement __init__

        # TODO: Initialize the sublayers  
        self.self_attn  = SelfAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        ) # Masked self-attention layer
        self.cross_attn = CrossAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        ) # Cross-attention layer
        self.ffn        = FeedForwardLayer(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        ) # Feed-forward network
        
        # pre-LN LayerNorms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # dropout after each sublayer
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, dec_key_padding_mask: Optional[torch.Tensor] = None, enc_key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the CrossAttentionDecoderLayer.
        Args:
            x (torch.Tensor): The input tensor. shape: (batch_size, seq_len, d_model)   
            enc_output (torch.Tensor): The encoder output. shape: (batch_size, seq_len, d_model)
            dec_key_padding_mask (Optional[torch.Tensor]): The padding mask for the decoder input. shape: (batch_size, seq_len)
            enc_key_padding_mask (Optional[torch.Tensor]): The padding mask for the encoder output. shape: (batch_size, seq_len')
            attn_mask (Optional[torch.Tensor]): The self-attention mask for the decoder input. shape: (seq_len, seq_len)
        Returns:
            x (torch.Tensor): The output tensor. shape: (batch_size, seq_len, d_model)
            self_attn_weights (torch.Tensor): The attention weights. shape: (batch_size, seq_len, seq_len)   
            cross_attn_weights (torch.Tensor): The attention weights. shape: (batch_size, seq_len, seq_len)    
        '''
        # TODO: Implement forward: Follow the figure in the writeup
        residual = x
        x_norm = self.norm1(x)
        x_attn, self_attn_weights  = self.self_attn(
            x=x_norm,
            key_padding_mask=dec_key_padding_mask,
            attn_mask=attn_mask,
        )
        x = residual + self.drop1(x_attn)

        residual = x
        x_norm = self.norm2(x)
        x_cross, cross_attn_weights = self.cross_attn(
            x=x_norm,
            y=enc_output,
            key_padding_mask=enc_key_padding_mask,
        )
        x = residual + self.drop2(x_cross)


        # TODO: Return the output tensor and attention weights    
        residual = x
        x_norm = self.norm3(x)
        x_ff = self.ffn(x_norm)
        output = residual + self.drop3(x_ff)
        return output, self_attn_weights, cross_attn_weights
## -------------------------------------------------------------------------------------------------    
