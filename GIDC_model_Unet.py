# -*- coding: utf-8 -*-
"""
PyTorch port of the GIDC DNN structure and ghost-imaging measurement process
reported in the paper:
Fei Wang et al. 'Far-field super-resolution ghost imaging with a deep neural
network constraint'. Light Sci Appl 11, 1 (2022).
https://doi.org/10.1038/s41377-021-00680-w
Please cite the paper if you find this code offers any help.

Original TensorFlow 1.x implementation by Fei Wang (WangFei_m@outlook.com).

Inputs (forward):
    inpt:     DGI result, shape [batch, 1, img_W, img_H]
    patterns: illumination patterns used as the measurement operator,
              shape [num_patterns, 1, img_W, img_H]

Outputs (forward):
    out_x: estimated image, shape [batch, 1, img_W, img_H]
    out_y: estimated single-pixel measurements, shape [batch, num_patterns]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

KERNEL = 5
PAD = KERNEL // 2  # 'SAME' padding for a stride-1, kernel-5 convolution


class _ConvBNAct(nn.Module):
    """stride-1 'SAME' conv -> batch norm -> activation."""

    def __init__(self, in_ch, out_ch, act='leaky'):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, KERNEL, stride=1, padding=PAD)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act == 'relu' else nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class _Down(nn.Module):
    """stride-2 'SAME' conv (down-sampling) -> batch norm -> leaky relu."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, KERNEL, stride=2, padding=PAD)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class _Up(nn.Module):
    """stride-2 transpose conv (2x up-sampling) -> batch norm -> leaky relu."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        # padding=2, output_padding=1 doubles the spatial size for kernel 5.
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, KERNEL, stride=2,
                                       padding=PAD, output_padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GIDCUNet(nn.Module):
    """U-Net with an embedded ghost-imaging physical (measurement) model.

    The network is never pre-trained: its weights are optimized from scratch
    for a single measurement set, with the physical model y = A x providing the
    only supervision.
    """

    def __init__(self):
        super().__init__()
        # ---- encoder ----
        self.conv0 = _ConvBNAct(1, 16)
        self.conv1 = _ConvBNAct(16, 16)
        self.conv1_1 = _ConvBNAct(16, 16)          # skip -> merge4
        self.pool1 = _Down(16, 16)
        self.conv2 = _ConvBNAct(16, 32)
        self.conv2_1 = _ConvBNAct(32, 32)          # skip -> merge3
        self.pool2 = _Down(32, 32)
        self.conv3 = _ConvBNAct(32, 64, act='relu')
        self.conv3_1 = _ConvBNAct(64, 64)          # skip -> merge2
        self.pool3 = _Down(64, 64)
        self.conv4 = _ConvBNAct(64, 128)
        self.conv4_1 = _ConvBNAct(128, 128)        # skip -> merge1
        self.pool4 = _Down(128, 128)
        self.conv5 = _ConvBNAct(128, 256)
        self.conv5_1 = _ConvBNAct(256, 256)
        # ---- decoder ----
        self.up6 = _Up(256, 128)
        self.conv6_1 = _ConvBNAct(256, 128)        # 128 (skip) + 128 (up)
        self.conv6_2 = _ConvBNAct(128, 128)
        self.up7 = _Up(128, 64)
        self.conv7_1 = _ConvBNAct(128, 64)         # 64 + 64
        self.conv7_2 = _ConvBNAct(64, 64)
        self.up8 = _Up(64, 32)
        self.conv8_1 = _ConvBNAct(64, 32)          # 32 + 32
        self.conv8_2 = _ConvBNAct(32, 32)
        self.up9 = _Up(32, 16)
        self.conv9_1 = _ConvBNAct(32, 16)          # 16 + 16
        self.conv9_2 = _ConvBNAct(16, 16)
        # ---- output ----
        self.conv10 = nn.Conv2d(16, 1, KERNEL, stride=1, padding=PAD)
        self.bn10 = nn.BatchNorm2d(1)

        self._init_weights()

    def _init_weights(self):
        # mirror the original truncated_normal(stddev=0.01) initialisation
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, inpt, patterns):
        # ---- encoder ----
        c0 = self.conv0(inpt)
        c1_1 = self.conv1_1(self.conv1(c0))
        c2_1 = self.conv2_1(self.conv2(self.pool1(c1_1)))
        c3_1 = self.conv3_1(self.conv3(self.pool2(c2_1)))
        c4_1 = self.conv4_1(self.conv4(self.pool3(c3_1)))
        c5_1 = self.conv5_1(self.conv5(self.pool4(c4_1)))
        # ---- decoder with skip connections ----
        m1 = torch.cat([c4_1, self.up6(c5_1)], dim=1)
        c6 = self.conv6_2(self.conv6_1(m1))
        m2 = torch.cat([c3_1, self.up7(c6)], dim=1)
        c7 = self.conv7_2(self.conv7_1(m2))
        m3 = torch.cat([c2_1, self.up8(c7)], dim=1)
        c8 = self.conv8_2(self.conv8_1(m3))
        m4 = torch.cat([c1_1, self.up9(c8)], dim=1)
        c9 = self.conv9_2(self.conv9_1(m4))
        out_x = torch.sigmoid(self.bn10(self.conv10(c9)))

        # ---- ghost-imaging measurement process (physical model) ----
        out_x = out_x / out_x.max()
        # full-image cross-correlation of the estimate with each pattern.
        out_y = F.conv2d(out_x, patterns).flatten(1)   # [batch, num_patterns]

        # normalisation (population statistics, matching tf.nn.moments)
        out_x = (out_x - out_x.mean()) / out_x.std(unbiased=False)
        out_y = (out_y - out_y.mean()) / out_y.std(unbiased=False)
        return out_x, out_y
