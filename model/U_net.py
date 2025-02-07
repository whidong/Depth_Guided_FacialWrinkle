from monai.networks.nets import UNet


# 모델 정의 (UNet)
weak_model = UNet(
    dimensions=2, 
    in_channels=3,     # RGB 입력
    out_channels=1,    # 텍스처 마스크 생성
    channels=(64, 128, 256, 512, 1024), 
    strides=(2, 2, 2, 2)
).to("cuda")