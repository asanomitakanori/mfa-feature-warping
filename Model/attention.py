
import torch
import torch.nn as nn


class DoubleConv2(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)


class MultiFrameAttention(nn.Module):
    """(Attention layer) use convolution instead of FC layer"""
    def __init__(self, in_channels, out_channels, sub_batch_size):
        super().__init__()
        self.Wq = DoubleConv2(in_channels, out_channels)
        self.Wk = DoubleConv2(in_channels, out_channels)
        self.Wv = DoubleConv2(in_channels, out_channels)
        self.softmax = nn.Softmax(dim =2)
        self.sub_batch_size = sub_batch_size

    def forward(self, input):
        temp = input.reshape(int(input.shape[0]/self.sub_batch_size), self.sub_batch_size, input.shape[1], input.shape[2], input.shape[3])
        V, B, C, H, W = temp.shape[0], temp.shape[1], temp.shape[2], temp.shape[3], temp.shape[4]
        query = self.Wq(input).reshape(V, B, -1)
        key = self.Wk(input).reshape(V, B, -1)
        value = self.Wv(input).reshape(V, B, -1)
        attention_weight = torch.matmul(query, key.permute(0, 2, 1)).reshape(V, B, -1)
        attention_weight = self.softmax(attention_weight)
        weighted_value= torch.matmul(attention_weight, value).reshape(V*B, C, H, W)
        return weighted_value + input

