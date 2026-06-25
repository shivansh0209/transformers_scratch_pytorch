import torch

def generate_positional_encoding(input_tensor):
    # input tensor shape: (batch_size, seq_len, d_model)
    batch_size, seq_len, d_model = input_tensor.size()
    positional_encoding = torch.zeros_like(input_tensor, dtype=torch.float32)

    for batch_idx in range(batch_size):
        for pos in range(seq_len):
            for i in range(int(d_model / 2)):
                denominator = 10000 ** ((2 * i) / d_model)
                positional_encoding[batch_idx, pos, 2 * i] = torch.sin(pos / denominator)
                positional_encoding[batch_idx, pos, 2 * i + 1] = torch.cos(pos / denominator)

    return positional_encoding

__all__ = ['generate_positional_encoding']