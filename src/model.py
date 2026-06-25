import torch
from torch import nn
from modules import MultiHeadAttention, CrossAttention, LayerNorm, FeedForward
from utils import generate_positional_encoding

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.layer_norm1 = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layer_norm2 = LayerNorm(d_model)

    def forward(self, x):
        attention_output = self.self_attention(x, hasMask=False)
        x = self.layer_norm1(x + attention_output)
        
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])

    def forward(self, input_tensor):
        input_tensor = input_tensor + generate_positional_encoding(input_tensor)
        for layer in self.layers:
            input_tensor = layer(input_tensor)
            
        return input_tensor
    

# class DecoderLayer(nn.Module):
