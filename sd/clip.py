import torch
import torch.nn as nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEMbedding(nn.Module):

    def __init__(self, n_vocab: int, n_embed: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.position_embedding = nn.Parameter(torch.zeros(1, n_tokens, n_embed))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (Batch_Size, Seq_len) -> (Batch_Size, Seq_len, Dim)
        x = self.token_embedding(tokens)

        # (Batch_Size, Seq_len, Dim) -> (Batch_Size, Seq_len, Dim)
        x =  x + self.position_embedding

        return x
    
class CLIPlayer(nn.Module):

    def __init__(self, n_heads: int, n_embed: int):
        super().__init__()

        self.layernor_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_heads, n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)
        self.linear_1 = nn.Linear(n_embed, n_embed * 4)
        self.linear_2 = nn.Linear(n_embed * 4, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Seq_len, Dim) -> (Batch_Size, Seq_len, Dim)
        residue = x

        # SELF ATTENTION
        x = self.layernorm_1(x)

        x = self.attention(x, casual_mask=True)

        x = x + residue

        # MLP
        residue = x

        x = self.layernorm_2(x)

        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702 * x)

        x = self.linear_2(x)

        x = x + residue

        return x

class CLIP(nn.Module):
    
    def __init__(self):
        self.embedding = CLIPEMbedding(49408, 768, 77)

        self.layers = nn.Module(
            CLIPlayer(12, 768) for i in range(12)
        )

        self.layernorm = nn.LayerNorm(768)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = tokens.type(torch.long)

        # tokens: (Batch_Size, Seq_len) -> (Batch_Size, Seq_len, Dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)
        
        output = self.layernorm(state)

        return output
    
