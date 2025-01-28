from monai.networks.nets import UNet


# 모델 정의 (UNet)
weak_model = UNet(
    dimensions=2, 
    in_channels=3,     # RGB 입력
    out_channels=1,    # 텍스처 마스크 생성
    channels=(64, 128, 256, 512, 1024), 
    strides=(2, 2, 2, 2)
).to("cuda")

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, channels):
        super(UNet, self).__init__()
        # 인코더 정의
        self.encoder1 = self.conv_block(in_channels, channels[0])
        self.encoder2 = self.conv_block(channels[0], channels[1])
        self.encoder3 = self.conv_block(channels[1], channels[2])
        self.encoder4 = self.conv_block(channels[2], channels[3])
        self.bottom = self.conv_block(channels[3], channels[4])

        # 디코더 정의
        self.decoder4 = self.upconv_block(channels[4], channels[3])
        self.decoder3 = self.upconv_block(channels[3], channels[2])
        self.decoder2 = self.upconv_block(channels[2], channels[1])
        self.decoder1 = self.upconv_block(channels[1], channels[0])

        # 최종 출력
        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)

        # 다운샘플링 레이어
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 인코더
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        bottom = self.bottom(self.pool(enc4))

        # 디코더
        dec4 = self.decoder4(torch.cat([F.interpolate(bottom, scale_factor=2, mode='bilinear', align_corners=True), enc4], dim=1))
        dec3 = self.decoder3(torch.cat([F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=True), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True), enc1], dim=1))

        # 최종 출력
        out = self.final_conv(dec1)
        return out

"""