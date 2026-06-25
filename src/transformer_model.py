import torch.nn as nn
from src.encoder_decoder import Encoder, Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, num_heads, d_ff, num_layers):
        super().__init__()
        # 1. Learnable Token Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        
        # 2. Structural blocks (Your custom components)
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers)
        
        # 3. LM Head to predict vocabulary IDs
        self.fc_out = nn.Linear(d_model, trg_vocab_size)

    def forward(self, source_tokens, target_tokens):
        # Pass raw token IDs through the respective embedding layers
        src_emb = self.src_embedding(source_tokens)
        trg_emb = self.trg_embedding(target_tokens)
        
        # Core forward pass pipeline
        encoder_output = self.encoder(src_emb)
        decoder_output = self.decoder(trg_emb, encoder_output)
        
        logits = self.fc_out(decoder_output)
        return logits # Return raw logits for PyTorch's CrossEntropyLoss