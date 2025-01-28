from monai.networks.nets import SwinUNETR

def create_swin_unetr(img_size=(1024, 1024), in_channels=1, out_channels=1, feature_size=48, use_checkpoint=True):
    """
    Swin UNETR 모델 생성 함수
    """
    return SwinUNETR(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint
    )
