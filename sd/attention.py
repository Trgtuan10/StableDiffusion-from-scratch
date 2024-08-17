import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

        def forward(self, x: torch.Tensor, casual_mask=False):
            # x: (Batch_Size, Length, dim)

            input_shape = x.shape

            batch_size, seq_len, d_embed = input_shape

            internim_shape = (batch_size, seq_len, self.n_heads, self.d_head)
            
            # x: (Batch_Size, Length, dim) -> (Batch_Size, Length, 3 * dim) -> (Batch_Size, Length, 3, dim)
            q, k, v = self.in_proj(x).chunk(3, dim=1)

            # (Batch_Size, Length, dim) -> (Batch_Size, Length, n_heads, d_head) -> (Batch_Size, n_heads, Length, d_head)
            q = q.view(internim_shape).transpose(1, 2)
            k = k.view(internim_shape).transpose(1, 2)
            v = v.view(internim_shape).transpose(1, 2)


            weight = q @ k.transpose(-1, -2)

            if casual_mask:
                #trianglar
                mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
                weight.masked_fill(mask, -torch.inf)

            weight /= math.sqrt(self.d_head)

            weight = F.softmax(weight, dim=-1)
        
            # (Batch_Size, n_heads, Length, d_head) -> (Batch_Size, Length, n_heads, d_head)
            output = weight @ v

            # (Batch_Size, Length, n_heads, d_head) -> (Batch_Size, Length, dim)
            output = output.transpose(1, 2).reshape(batch_size, seq_len, d_embed)

            output = self.out_proj(output)

            return output
        
class CrossAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)

        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, latent: torch.Tensor, context: torch.Tensor):
        # latent: (Batch_Size, seq_length_q, dim_q)
        # context: (Batch_Size, seq_length_k_v, dim_k_v) = (batch_size, 77, 768)

        input_shape = latent.shape
        batch_size, sequence_length, d_embed = input_shape

        internim_shape = (batch_size, -1, self.n_heads, self.d_head)

        #Multiquery by Wq
        q = self.q_proj(latent)
        k = self.k_proj(context)
        v = self.v_proj(context)

        weight = q @ k.transpose(-1, -2)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        output = weight @ v

        output = output.transpose(1, 2).reshape(batch_size, sequence_length, d_embed)

        output = self.out_proj(output)

        return output

        