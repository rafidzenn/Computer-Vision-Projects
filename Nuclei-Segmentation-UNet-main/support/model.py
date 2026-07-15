"""A small, standard U-Net for binary nuclei segmentation.

Kept deliberately simple for the proof-of-concept: 4 down / 4 up levels,
BatchNorm + ReLU, transposed-conv upsampling. Input spatial size must be
divisible by 16 (e.g. 128, 256).
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(conv -> BN -> ReLU) x 2."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # encoder
        self.d1 = DoubleConv(in_ch, base)
        self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.d4 = DoubleConv(base * 4, base * 8)
        self.bottleneck = DoubleConv(base * 8, base * 16)

        # decoder
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.u4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.u3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.u2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.u1 = DoubleConv(base * 2, base)

        self.head = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))
        bn = self.bottleneck(self.pool(c4))

        x = self.u4(torch.cat([self.up4(bn), c4], dim=1))
        x = self.u3(torch.cat([self.up3(x), c3], dim=1))
        x = self.u2(torch.cat([self.up2(x), c2], dim=1))
        x = self.u1(torch.cat([self.up1(x), c1], dim=1))
        return self.head(x)  # raw logits (apply sigmoid at loss/inference time)
