import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    
    def __init__(self, n_embd: int):
        super().__init__
        self.linear_1 = nn.Linear(n_embd, 4*n_embd)
        self.linear_2 = nn.Linear(4*n_embd, 4*n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 320)
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))

        # x: (1, 1280)
        return x
    
class UNET_ResidualBlock(nn.Module):

    def __init__(self, in_channel: int, out_channel: int, n_time=1280):
        super().__init__()
        self.group_norm_feature = nn.GroupNorm(32, in_channel)
        self.conv_feature = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channel)

        self.groupnorm_merge = nn.GroupNorm(32, out_channel)
        self.conv_merge = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)

        if in_channel == out_channel:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)

    def forward(self, feature: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # feature: (Batch_Size, in_channel, Height, Width)
        # time (1, 1280)
        residue = feature

        feature = self.group_norm_feature(feature)

        feature = F.silu(feature)

        feature = self.conv_feature(feature)

        time = F.silu(time)

        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.groupnorm_merge(merged)

        merged = F.silu(merged)

        merged = self.conv_merge(merged)

        return merged + self.residual_layer(residue)
    
class UNET_AttentionBlock(nn.Module):

    def __init__(self, n_head: int, n_embed: int, d_context: int):
        super().__init__()
        channels = n_head * n_embed

        self.groupnorm = nn.GroupNorm(32, channels)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, n_embed, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x, context):
        # x: (Batch_Size, channels, Height, Width)
        # context: (Batch_Size, seg_len, dim)
        
        residue_long = x

        x = self.groupnorm(x)

        x = self.conv_input(x)

        n, c, h, w = x.shape

        # x: (Batch_Size, channels, Height, Width) -> (Batch_Size, channels, Height * Width)
        x = x.view(n, c, h * w)

        # x: (Batch_Size, channels, Height, Width) -> (Batch_Size, Height * Width, channels)
        x = x.transpose(1, 2)

        # Normlization + Self Attention with skip connection

        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x = x + residue_short

        residue_short = x

        # Normlization + Cross Attention with skip connection
        x = self.layernorm_2(x)

        self.attention_2(x, context)

        x = x + residue_short

        x = self.layernorm_3(x)

        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)

        x = x * F.gelu(gate)

        x = self.linear_geglu_2(x)

        x = x + residue_short

        # x: (Batch_Size, Height * Width, channels) -> (Batch_Size, channels, Height * Width)
        x = x.transpose(-1, -2)

        # x: (Batch_Size, channels, Height * Width) -> (Batch_Size, channels, Height, Width)   
        x = x.view(n, c, h, w)

        # x: (Batch_Size, channels, Height, Width) -> (Batch_Size, channels, Height, Width)
        x = self.conv_output(x)

        return x + residue_long

class Upsample(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, channels, Height, Width) -> (Batch_Size, channels, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)

        return x

class SwitchSequential(nn.Sequential):

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            elif isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            else:
                x = layer(x)

        return x

class UNET(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList([
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            

            # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 16, Width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            

            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 32, Width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])

        self.bottle_neck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),

            UNET_AttentionBlock(8, 160),

            UNET_ResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList([
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 32, Width / 32) 
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            
            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1920, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),
            
            # (Batch_Size, 1920, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 1280, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 960, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
            
            # (Batch_Size, 960, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

class UNET_OutputLayer(nn.Module):

    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_channel)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)     

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, in_channel, Height / 8, Width / 8) -> (Batch_Size, out_channel, Height / 8, Width / 8)
        x = self.group_norm(x)

        x = F.silu(x)

        x = self.conv(x)

        return x   

class Diffusion(nn.Module):

    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

        def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
            #latent: (batch_size, 4, height/8, width/8)
            #context: (batch_size, seg_len, dim)
            #time: (1, 320)

            # (1, 320) -> (1, 1280)
            time = self.time_embedding(time)

            # (batch_size, 4, height/8, width/8) -> (batch_size, 320, height/8, width/8)
            output = self.unet(latent, context, time)

            # (batch_size, 320, height/8, width/8) -> (batch_size, 4, height/8, width/8)
            output = self.final(output)

            return output