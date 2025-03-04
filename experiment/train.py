import torch
from model import create_unet, create_swin_unetr

# 모델 생성
unet_model = create_unet()
swin_unetr_model = create_swin_unetr()

# 입력 데이터
input_tensor = torch.rand((1, 1, 1024, 1024))

# 출력 확인
unet_output = unet_model(input_tensor)
swin_output = swin_unetr_model(input_tensor)

print(f"UNet Output shape: {unet_output.shape}")
print(f"Swin UNETR Output shape: {swin_output.shape}")
