import torch
import torch.nn as nn
import math

# class Cross_attention(nn.Module):
#     def __init__(self, in_channel, n_head=1, norm_groups=16):
#         super().__init__()
#         self.n_head = n_head
#         self.norm_A = nn.GroupNorm(norm_groups, in_channel)
#         self.norm_B = nn.GroupNorm(norm_groups, in_channel)
#         self.qkv_A = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
#         self.out_A = nn.Conv2d(in_channel, in_channel, 1)
#
#         self.qkv_B = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
#         self.out_B = nn.Conv2d(in_channel, in_channel, 1)
#
#         self.out_F = nn.Conv2d(in_channel*2, in_channel, 1)
#
#     def forward(self, x_A, x_B):
#         batch, channel, height, width = x_A.shape
#
#         n_head = self.n_head
#         head_dim = channel // n_head
#
#         x_A = self.norm_A(x_A)
#         qkv_A = self.qkv_A(x_A).view(batch, n_head, head_dim * 3, height, width)
#         query_A, key_A, value_A = qkv_A.chunk(3, dim=2)
#
#         x_B = self.norm_B(x_B)
#         qkv_B = self.qkv_B(x_B).view(batch, n_head, head_dim * 3, height, width)
#         query_B, key_B, value_B = qkv_B.chunk(3, dim=2)
#
#         attn_A = torch.einsum(
#             "bnchw, bncyx -> bnhwyx", query_B, key_A
#         ).contiguous() / math.sqrt(channel)
#         attn_A = attn_A.view(batch, n_head, height, width, -1)
#         attn_A = torch.softmax(attn_A, -1)
#         attn_A = attn_A.view(batch, n_head, height, width, height, width)
#
#         out_A = torch.einsum("bnhwyx, bncyx -> bnchw", attn_A, value_A).contiguous()
#         out_A = self.out_A(out_A.view(batch, channel, height, width))
#         out_A = out_A + x_A
#
#         attn_B = torch.einsum(
#             "bnchw, bncyx -> bnhwyx", query_A, key_B
#         ).contiguous() / math.sqrt(channel)
#         attn_B = attn_B.view(batch, n_head, height, width, -1)
#         attn_B = torch.softmax(attn_B, -1)
#         attn_B = attn_B.view(batch, n_head, height, width, height, width)
#
#         out_B = torch.einsum("bnhwyx, bncyx -> bnchw", attn_B, value_B).contiguous()
#         out_B = self.out_B(out_B.view(batch, channel, height, width))
#         out_B = out_B + x_B
#
#         return self.out_F(torch.cat([out_A, out_B], 1))

class Cross_attention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()
        self.conv = nn.Conv2d(in_channel*2, in_channel, kernel_size=1)

    def forward(self, vi, ir):
        return self.conv(torch.cat((vi, ir), dim=1))