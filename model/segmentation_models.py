from monai.networks.nets import SwinUNETR, UNet as MONAI_UNet
from model.Custom_UNet import UNet as CustomUNet

def create_model(model_type="swin_unetr", img_size=(96, 96, 96), in_channels=1, out_channels=1, feature_size=48, use_checkpoint=True):
    """
    모델 생성 함수
    Parameters:
        - model_type (str): "swin_unetr" 또는 "unet" 중 선택
        - img_size (tuple): 입력 이미지 크기
        - in_channels (int): 입력 채널 수
        - out_channels (int): 출력 채널 수 (클래스 수)
        - feature_size (int): Swin UNETR에서 기본 채널 크기
        - use_checkpoint (bool): Swin UNETR의 메모리 최적화 옵션
    Returns:
        - model: 선택한 세그멘테이션 모델
    """
    if model_type.lower() == "swin_unetr":
        return SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            spatial_dims = 2
        )
    elif model_type.lower() == "unet":
        return MONAI_UNet(
            spatial_dims=2,  # 2D 데이터
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(64, 128, 256, 512, 1024),
            strides=(2, 2, 2, 2),
            num_res_units=0
        )
    elif model_type.lower() == "custom_unet":
        return CustomUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(64, 128, 256, 512, 1024),
            bilinear=True
        )
    else:
        raise ValueError("Invalid model_type. Choose either 'swin_unetr' or 'unet'.")


