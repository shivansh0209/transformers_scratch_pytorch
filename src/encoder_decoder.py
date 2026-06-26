from torch import nn
from src.modules import MultiHeadAttention, CrossAttention, LayerNorm, FeedForward
from src.utils import generate_positional_encoding

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
        input_tensor = input_tensor
        for layer in self.layers:
            input_tensor = layer(input_tensor)
            
        return input_tensor
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        # 1. Masked Self Attention (Targets look only at past targets)
        self.masked_attention = MultiHeadAttention(d_model, num_heads)
        self.layer_norm1 = LayerNorm(d_model)
        
        # 2. Cross Attention (Targets look at Encoder outputs)
        self.cross_attention = CrossAttention(d_model, num_heads)
        self.layer_norm2 = LayerNorm(d_model)
        
        # 3. Feed Forward Network
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layer_norm3 = LayerNorm(d_model)

    def forward(self, decoder_input, encoder_output):
        # Sub-layer 1: Masked Self-Attention
        masked_attn_out = self.masked_attention(decoder_input, hasMask=True)
        x = self.layer_norm1(decoder_input + masked_attn_out)
        
        # Sub-layer 2: Cross-Attention (Pass both decoder and encoder states)
        cross_attn_out = self.cross_attention(x, encoder_output)
        x = self.layer_norm2(x + cross_attn_out)
        
        # Sub-layer 3: Feed Forward
        ff_out = self.feed_forward(x)
        x = self.layer_norm3(x + ff_out)
        
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers=6):
        super().__init__()
        # Stack multiple identical decoder layers
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])

    def forward(self, decoder_input, encoder_output):
        # 1. Add positional encoding to the target sequence once at the start
        decoder_input = decoder_input
        
        # 2. Pass sequentially through all stacked decoder layers
        for layer in self.layers:
            decoder_input = layer(decoder_input, encoder_output)
            
        return decoder_input