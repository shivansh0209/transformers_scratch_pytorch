import torch
import torch.nn as nn
from encodr_decoder import Encoder, Decoder

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size):
        super().__init__()
        # 1. Core structural blocks
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers)
        
        # 2. Final Output Projection Layer (The LM Head)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, source_tokens, target_tokens):
        # Step 1: Run the source text through the full Encoder stack
        encoder_output = self.encoder(source_tokens)
        
        # Step 2: Run the target text + encoder output through the Decoder stack
        decoder_output = self.decoder(target_tokens, encoder_output)
        
        # Step 3: Project decoder features to vocabulary size
        logits = self.fc_out(decoder_output)
        
        # Step 4: Apply Softmax to get probability distribution over the vocab
        output_probabilities = torch.softmax(logits, dim=-1)
        
        return output_probabilities