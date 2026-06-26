import torch
import torch.nn as nn
from src.utils import rotational_positional_encoding
# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, num_heads=1):
#         super().__init__()
#         assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_k = d_model // num_heads  # Split the dimension across heads
        
#         # MUST use nn.ModuleList so PyTorch tracks the parameters
#         self.W_q = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
#         self.W_k = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
#         self.W_v = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        
#         # Concat size will be (num_heads * d_k) which equals d_model
#         self.merger_matrix = nn.Linear(d_model, d_model)

#     def forward(self, input_tensor, hasMask=False):
#         multi_output = []
#         mask = None

#         if hasMask:
#             mask = torch.tril(torch.ones(input_tensor.size(1), input_tensor.size(1)))
#             mask = mask.unsqueeze(0).repeat(input_tensor.size(0), 1, 1)

#         for i in range(self.num_heads):
#             Q = self.W_q[i](input_tensor)
#             K = self.W_k[i](input_tensor)
#             V = self.W_v[i](input_tensor)
#             attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
#             if hasMask:
#                 attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
#             attention_weights = torch.softmax(attention_scores, dim=-1)
#             output = torch.matmul(attention_weights, V)
#             multi_output.append(output)

#         output = torch.cat(multi_output, dim=-1)
#         output = self.merger_matrix(output)
#         return output

# RETROSPECTIVE: Optimization point 2
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Split dimension across heads
        
        # OPTIMIZATION: Instead of lists of small linears, use one big linear layer for each matrix
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        self.merger_matrix = nn.Linear(d_model, d_model)

    def forward(self, input_tensor, hasMask=False):
        batch_size, seq_len, _ = input_tensor.size()

        # 1. Project all heads at once 
        # Shape changes from (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        q_all = self.W_q(input_tensor)
        k_all = self.W_k(input_tensor)
        v_all = self.W_v(input_tensor)

        # 2. Reshape and transpose to isolate the heads
        # Split d_model into (num_heads, d_k), then swap seq_len and num_heads
        # Final shape for Q, K, V: (batch_size, num_heads, seq_len, d_k)
        Q = q_all.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = k_all.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = v_all.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 3. Compute Attention Scores for all heads in parallel
        # Q matrix multiplied by transposed K matrix across the last two dimensions
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        Q = rotational_positional_encoding(Q)
        K = rotational_positional_encoding(K)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        # 4. Mask application using broadcasting (no repeat needed!)
        if hasMask:
            # Mask shape: (1, 1, seq_len, seq_len) to broadcast across batch and num_heads dimensions
            mask = torch.tril(torch.ones(seq_len, seq_len, device=input_tensor.device)).unsqueeze(0).unsqueeze(0)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # 5. Softmax and Weighted Values
        attention_weights = torch.softmax(attention_scores, dim=-1)
        # Shape: (batch_size, num_heads, seq_len, d_k)
        output = torch.matmul(attention_weights, V)

        # 6. Concat and Merge Heads back together
        # Swap num_heads and seq_len back, then flatten the last two dimensions
        # Shape changes: (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, num_heads, d_k) -> (batch_size, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.merger_matrix(output)


class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads=1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
    
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        self.merger_matrix = nn.Linear(d_model, d_model)

    def forward(self, query_tensor, key_value_tensor):
        batch_size, seq_len_q, _ = query_tensor.size()
        _, seq_len_kv, _ = key_value_tensor.size()

        Q = self.W_q(query_tensor)
        K = self.W_k(key_value_tensor)
        V = self.W_v(key_value_tensor)

        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_kv, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_kv, self.num_heads, self.d_k).transpose(1, 2)

        # Scale by the split head dimension d_k, not the full d_model
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        return self.merger_matrix(output)
    


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
        mean = torch.mean(input_tensor, dim=-1, keepdim=True)
        variance = torch.var(input_tensor, dim=-1, keepdim=True, unbiased=False)
        nrm = (input_tensor - mean) / torch.sqrt(variance + self.eps)
        output = self.gamma * nrm + self.beta
        return output

        # output = torch.zeros_like(input_tensor)

        # for b in range(batch_size):
        #     for s in range(seq_len):
        #         # 1. Gather all d_model features for the current token
        #         token_features = input_tensor[b, s, :]
                
        #         # 2. Calculate mean and variance for this single token
        #         mean = torch.mean(token_features)
        #         variance = torch.var(token_features, unbiased=False)
                
        #         # 3. Normalize, scale with gamma, and shift with beta
        #         # (eps prevents division by zero)
        #         normalized = (token_features - mean) / torch.sqrt(variance + self.eps)
        #         output[b, s, :] = self.gamma * normalized + self.beta

        # return output
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, input_tensor):
        return self.linear2(self.relu(self.linear1(input_tensor)))
    
__all__ = ['MultiHeadAttention', 'CrossAttention', 'LayerNorm', 'FeedForward']