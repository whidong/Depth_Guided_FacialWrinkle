import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    conv -> BN -> ReLU -> conv -> BN -> ReLU 
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels

        # print(f"in {in_channels}, out {out_channels}, mid {mid_channels}")
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

class Down(nn.Module):
    """
    Down sampling: MaxPool(stride=2) -> DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )
        
    def forward(self, x):
        # print(f"Down before x : {x.shape}")
        return self.pool_conv(x)

class Up(nn.Module):
    """
    Upsampling : (Bilinear Interpolation or ConvTranspose2d) -> skip 연결 -> DoubleConv
    - Bilinear
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        """
        in_channels: cat (ex. 512 + 256)
        out_channels: final channel (DoubleConv 이후)
        """
        super().__init__()
        
        if not bilinear:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        
    def forward(self, x1, x2):
        """
        x1: Upsampling output (lower resolution)
        x2: Skip connection feature (result of encoder, higher resolution)
        """
        # print(f"UP before x1 : {x1.shape}, x2 : {x2.shape}")
        # bilinear upsampling
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # print(f"UP After x1 :{x1.shape}, x2 : {x2.shape}")
        # channel cat
        x = torch.cat([x2, x1], dim=1)
        print(f"concat x :{x.shape}")
        # DoubleConv
        return self.conv(x)

class UNet(nn.Module):
    """
    - channels = (64, 128, 256, 512, 1024)
    - in_channels: 입력 채널 (ex. 3)
    - out_channels: 출력 채널 (ex. 1)
    """
    def __init__(self, in_channels, out_channels, channels=(64, 128, 256, 512, 1024), bilinear=True):
        super(UNet, self).__init__()
        
        # (Optional) Input preprocessing Block
        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        self.bilinear = bilinear
        # Encoder (Down)
        self.enc1 = DoubleConv(channels[0], channels[0])     # 64->64
        self.enc2 = Down(channels[0], channels[1])           # 64->128
        self.enc3 = Down(channels[1], channels[2])           #128->256
        self.enc4 = Down(channels[2], channels[3])           #256->512
        factor = 2 if bilinear else 1       
        self.bottom = Down(channels[3], channels[4] // factor)         #512->1024
        # Decoder (Up)
        # skip connect: cat([upsampling x, enc4]) → channel: (512 + 512) = 1536
        # After DoubleConv out_channels = 512
        self.dec4 = Up(channels[4], channels[3] // factor, bilinear)  # 512+512 ->1024
        self.dec3 = Up(channels[3], channels[2] // factor, bilinear)  # 256+256 ->512
        self.dec2 = Up(channels[2], channels[1] // factor, bilinear)  # 128+128 ->256
        self.dec1 = Up(channels[1], channels[0], bilinear=bilinear)  # 64+64  ->64
        
        # 최종 출력
        self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        # (Optional) input preprocessing
        x0 = self.input_block(x)      # [N, 64, H, W]
        
        # Encoder
        x1 = self.enc1(x0)            # [N, 64, H, W]
        x2 = self.enc2(x1)            # [N,128, H/2, W/2]
        x3 = self.enc3(x2)            # [N,256, H/4, W/4]
        x4 = self.enc4(x3)            # [N,512, H/8, W/8]
        x5 = self.bottom(x4)          # [N,1024,H/16,W/16]

        # Decoder
        d4 = self.dec4(x5, x4)        # [N,512, H/8, W/8]
        d3 = self.dec3(d4, x3)        # [N,256, H/4, W/4]
        d2 = self.dec2(d3, x2)        # [N,128, H/2, W/2]
        d1 = self.dec1(d2, x1)        # [N,64,  H,   W  ]

        # final output
        out = self.out_conv(d1)       # [N, out_channels, H, W]
        return out
    
    