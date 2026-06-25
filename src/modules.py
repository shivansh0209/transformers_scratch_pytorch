import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Split the dimension across heads
        
        # MUST use nn.ModuleList so PyTorch tracks the parameters
        self.W_q = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.W_k = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.W_v = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        
        # Concat size will be (num_heads * d_k) which equals d_model
        self.merger_matrix = nn.Linear(d_model, d_model)

    def forward(self, input_tensor, hasMask=False):
        multi_output = []
        mask = None

        if hasMask:
            mask = torch.tril(torch.ones(input_tensor.size(1), input_tensor.size(1)))
            mask = mask.unsqueeze(0).repeat(input_tensor.size(0), 1, 1)

        for i in range(self.num_heads):
            Q = self.W_q[i](input_tensor)
            K = self.W_k[i](input_tensor)
            V = self.W_v[i](input_tensor)
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
            if hasMask:
                attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            attention_weights = torch.softmax(attention_scores, dim=-1)
            output = torch.matmul(attention_weights, V)
            multi_output.append(output)

        output = torch.cat(multi_output, dim=-1)
        output = self.merger_matrix(output)
        return output
    
class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads=1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Split the dimension across heads
        
        # MUST use nn.ModuleList so PyTorch tracks the parameters
        self.W_q = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.W_k = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.W_v = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        
        # Concat size will be (num_heads * d_k) which equals d_model
        self.merger_matrix = nn.Linear(d_model, d_model)

    def forward(self, query_tensor, key_value_tensor):
        multi_output = []

        for i in range(self.num_heads):
            Q = self.W_q[i](query_tensor)
            K = self.W_k[i](key_value_tensor)
            V = self.W_v[i](key_value_tensor)

            # Scale by the split head dimension d_k, not the full d_model
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
            attention_weights = torch.softmax(attention_scores, dim=-1)

            output = torch.matmul(attention_weights, V)
            multi_output.append(output)

        output = torch.cat(multi_output, dim=-1)
        output = self.merger_matrix(output)
        return output
    


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        
        # Learnable parameters (initialized to 1s and 0s)
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, input_tensor):
        # input_tensor shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = input_tensor.size()
        
        # Create an empty tensor to store the normalized output
        output = torch.zeros_like(input_tensor)

        for b in range(batch_size):
            for s in range(seq_len):
                # 1. Gather all d_model features for the current token
                token_features = input_tensor[b, s, :]
                
                # 2. Calculate mean and variance for this single token
                mean = torch.mean(token_features)
                variance = torch.var(token_features, unbiased=False)
                
                # 3. Normalize, scale with gamma, and shift with beta
                # (eps prevents division by zero)
                normalized = (token_features - mean) / torch.sqrt(variance + self.eps)
                output[b, s, :] = self.gamma * normalized + self.beta

        return output
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, input_tensor):
        return self.linear2(self.relu(self.linear1(input_tensor)))
    
__all__ = ['MultiHeadAttention', 'CrossAttention', 'LayerNorm', 'FeedForward']