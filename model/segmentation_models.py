from monai.networks.nets import SwinUNETR, UNet as MONAI_UNet
from model.Custom_UNet import UNet as CustomUNet
from model.Custom_swin_unetr import SwinUNETR as Custom_SwinUNETR
from .simMIM import SwinTransformerForSimMIM, SimMIM
import segmentation_models_pytorch as smp

def create_model(model_type="swin_unetr", img_size=(1024, 1024), depth = (2, 2, 6, 2), in_channels=3, out_channels=1, feature_size=48, use_checkpoint=True, use_v2=True, pretrain = False, pretrain_path = "None"):
    """
    Create model
    Parameters:
        - model_type (str): swin_unetr, custom_unetr, imagenet_unet, unet, custom_unet
        - img_size (tuple): Input image resolution
        - in_channels (int): input channel
        - out_channels (int): output channel (class)
        - feature_size (int): Swin UNETR transformer embedding feature size
        - depth (sequencial: int) : Swin UNETR number of swinTransformer block
        - pretrain (bool) : Swin UNETR use pretrained imageNet SwinTransformerV2
        - pretrain_path (str): path of pretrain ckpt file
        - use_checkpoint (bool): Swin UNETR memory optimize
        - use_v2 (bool): select Swin UNETR backbone SwinTransformer or SwinTransformerV2
    Returns:
        - model: select segmentation model
    """
    # Swin UNETR model
    if model_type.lower() == "swin_unetr":
        return SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            spatial_dims = 2,
            depths = (2, 2, 2, 2),
            num_heads = (3, 6, 12, 24),
        )
    elif model_type.lower() == "custom_unetr":
        return Custom_SwinUNETR(
            img_size=img_size,
            in_channels = in_channels,
            out_channels = out_channels,
            feature_size = feature_size,
            use_checkpoint = use_checkpoint,
            spatial_dims = 2,
            depths = depth,
            num_heads = (3, 6, 12, 24),
            use_v2 = use_v2,
            pretrain = pretrain,
            pretrain_path = pretrain_path
        )
    elif model_type.lower() == "maked_swin":
        encoder = SwinTransformerForSimMIM(
            img_size=img_size,
            in_chans = in_channels,
            embed_dim = feature_size,
            use_checkpoint = use_checkpoint,
            spatial_dims = 2,
            depths = (2, 2, 2, 2),
            num_heads = (3, 6, 12, 24),
        )
        encoder_stride = 64
        model = SimMIM(encoder, encoder_stride)
        return model
    # UNet model
    elif model_type.lower() == "imagenet_unet":
        return smp.Unet(
            encoder_name = "resnet50",
            encoder_weight = "imagenet",
            decoder_interpolation_mode ="bilinear",
            decoder_channels = (256, 128, 64, 32, 16),
            in_channels = in_channels,
            classes = out_channels,
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


