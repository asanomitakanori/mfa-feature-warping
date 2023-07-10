""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch
from Model.attention import MultiFrameAttention
from Model.model_parts import *


class MFAmodel(nn.Module):
    def __init__(self, n_channels, n_classes, num, sub_batch, bilinear=True):
        super(MFAmodel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, num)
        self.down1 = Down(num, 2 * num)
        self.down2 = Down(2 * num, 4 * num)
        self.down3 = Down(4 * num, 8 * num)
        factor = 2 if bilinear else 1
        self.down4 = Down(8 * num, 16 * num // factor)
        self.up1 = Up(16 * num, 8 * num // factor, bilinear)
        self.up2 = Up(8 * num, 4 * num // factor, bilinear)
        self.up3 = Up(4 * num, 2 * num // factor, bilinear)
        self.up4 = Up(2 * num, num, bilinear)
        self.outc = OutConv(num, n_classes)

        self.mfa = MultiFrameAttention(
            16 * num // factor, 16 * num // factor, sub_batch
        )
        self.forward_flow = Flow_estimation(16 * num // factor, sub_batch)
        self.backward_flow = Backlow_estimation(16 * num // factor, sub_batch)
        self.sub_batch = sub_batch

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        forward_flow, forwarp_feature = self.forward_flow(x5)
        forwarp_feature = self.mfa(forwarp_feature)
        backward_flow, backwarp_feature = self.backward_flow(x5, forwarp_feature)
        x = self.up1(backwarp_feature, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits, [x5, forward_flow, backward_flow]
