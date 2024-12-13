from monai.networks.nets import UNet


# 모델 정의 (UNet)
weak_model = UNet(
    dimensions=2, 
    in_channels=3,     # RGB 입력
    out_channels=1,    # 텍스처 마스크 생성
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2)
).to("cuda")

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from torchvision.transforms import Resize


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, feature_sizes=[64, 128, 256, 512, 1024]):
        super(UNet, self).__init__()

        # Encoder
        self.encoders = nn.ModuleList([
            self._block(in_channels if i == 0 else feature_sizes[i-1], feature_sizes[i])
            for i in range(len(feature_sizes))
        ])
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=2) for _ in range(len(feature_sizes) - 1)])

        # Bottleneck
        self.bottleneck = self._block(feature_sizes[-1], feature_sizes[-1] * 2)

        # Decoder
        self.decoders = nn.ModuleList([
            self._block(feature_sizes[i] * 2, feature_sizes[i])
            for i in reversed(range(len(feature_sizes) - 1))
        ])
        self.upsamples = nn.ModuleList([
            nn.ConvTranspose2d(feature_sizes[i] * 2, feature_sizes[i], kernel_size=2, stride=2)
            for i in reversed(range(len(feature_sizes) - 1))
        ])

        # Final layer
        self.final_conv = nn.Conv2d(feature_sizes[0], out_channels, kernel_size=1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc_results = []

        # Encoder
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            enc_results.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for decoder, upsample, enc_result in zip(self.decoders, self.upsamples, reversed(enc_results)):
            x = upsample(x)
            x = torch.cat((x, enc_result), dim=1)
            x = decoder(x)

        return self.final_conv(x)
"""