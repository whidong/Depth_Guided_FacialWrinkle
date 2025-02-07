from __future__ import annotations

import itertools
from collections.abc import Sequence

import sys
sys.path.append("../")  # 프로젝트 루트를 경로에 추가
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from typing_extensions import Final

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from monai.utils.deprecate_utils import deprecated_arg
from monai.networks.utils import copy_model_state
from model.swin_transformer_v2 import SwinTransformerV2, filter_swinunetr
from model.swin_transformer import SwinTransformer

class SwinUNETR(nn.Module):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """

    patch_size: Final[int] = 4

    @deprecated_arg(
        name="img_size",
        since="1.3",
        removed="1.5",
        msg_suffix="The img_size argument is not required anymore and "
        "checks on the input size are run during forward().",
    )
    def __init__(
        self,
        img_size: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_checkpoint: bool = False,
        spatial_dims: int = 2,
        use_v2: bool = True,
        pretrain = False,
        pretrain_path: str = "None"
    ) -> None:
        """
        Args:
            img_size: spatial dimension of input image.
                This argument is only used for checking that the input image size is divisible by the patch size.
                The tensor passed to forward() can have a dynamic shape as long as its spatial dimensions are divisible by 2**5.
                It will be removed in an upcoming version.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).

        Examples::

            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)

            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))

            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETR(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)

        """

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = 4
        window_size = 16

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        self._check_input_size(img_size)

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        if use_v2 == True:
            self.swinViT = SwinTransformerV2(
                in_chans=in_channels,
                embed_dim=feature_size,
                window_size=window_size,
                patch_size=patch_size,
                depths=depths,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dropout_path_rate,
                norm_layer=nn.LayerNorm,
                use_checkpoint=use_checkpoint,
            )
        elif use_v2 == False:
            self.swinViT = SwinTransformer(
                in_chans=in_channels,
                embed_dim=feature_size,
                window_size=window_size,
                patch_size=patch_size,
                depths=depths,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dropout_path_rate,
                norm_layer=nn.LayerNorm,
                use_checkpoint=use_checkpoint,
            )
        if pretrain:
            print(f"Loading pretrained weights from {pretrain_path}...")
            checkpoint = torch.load(pretrain_path, map_location="cpu")
            ckpt_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

            if "norm.weight" in ckpt_dict:
                ckpt_dict["layers.3.downsample.norm.weight"] = ckpt_dict.pop("norm.weight")

            if "norm.bias" in ckpt_dict:
                ckpt_dict["layers.3.downsample.norm.bias"] = ckpt_dict.pop("norm.bias")

            dst_dict, loaded, not_loaded = copy_model_state(self.swinViT, ckpt_dict, filter_func=filter_swinunetr)
            self.swinViT.load_state_dict(dst_dict, strict=False)
            print(f"Pretrained weights loaded successfully. ({len(loaded)} keys loaded, {len(not_loaded)} keys not loaded)")

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=4,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

    def filter_swinunetr(key, value):
    # 2) 만약 norm.weight/bias 중 "patch_embed"가 아닌 것은 스킵
    #    ex) "norm.weight" / "layers.3.downsample.norm.weight" / "swinViT.norm.weight" 등
    # if "norm.weight" in key or "norm.bias" in key:
    #     return None
        
        skip_list = [
        "head.",
        "log_clamp_val",
        "attn_mask",
        "layers.3.blocks.1.attn.relative_coords_table",
        "layers.3.blocks.1.attn.relative_position_index",
        "layers.3.blocks.0.attn.relative_coords_table",
        "layers.3.blocks.0.attn.relative_position_index",
        "layers.3.downsample.norm.weight",
        "layers.3.downsample.norm.bias"
        ]
        
        if any(s in key for s in skip_list):
            return None
        # 2) 나머지는 "swinViT." + key 로 rename
        new_key = f"swinViT.{key}"
        return (new_key, value)

    def load_from(self, weights):
        with torch.no_grad():
            self.swinViT.patch_embed.proj.weight.copy_(weights["state_dict"]["module.patch_embed.proj.weight"])
            self.swinViT.patch_embed.proj.bias.copy_(weights["state_dict"]["module.patch_embed.proj.bias"])
            for bname, block in self.swinViT.layers1[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers1")
            self.swinViT.layers1[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.reduction.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers2[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers2")
            self.swinViT.layers2[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.reduction.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers3[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers3")
            self.swinViT.layers3[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.reduction.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers4[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers4")
            self.swinViT.layers4[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.reduction.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.bias"]
            )

    @torch.jit.unused
    def _check_input_size(self, spatial_shape):
        img_size = np.array(spatial_shape)
        remainder = (img_size % np.power(self.patch_size, 5)) > 0
        if remainder.any():
            wrong_dims = (np.where(remainder)[0] + 2).tolist()
            raise ValueError(
                f"spatial dimensions {wrong_dims} of input image (spatial shape: {spatial_shape})"
                f" must be divisible by {self.patch_size}**5."
            )
    
    def forward(self, x_in):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._check_input_size(x_in.shape[2:])
        hidden_states_out = self.swinViT(x_in)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        return logits

    # check parameter shape
    """
    def forward(self, x_in):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._check_input_size(x_in.shape[2:])
        
        # 1) 인코더 출력
        hidden_states_out = self.swinViT(x_in)
        print(f"[DEBUG] hidden_states_out shapes:")
        for i, h in enumerate(hidden_states_out):
            print(f"  hidden_states_out[{i}].shape = {h.shape}")

        enc0 = self.encoder1(x_in)
        print(f"[DEBUG] enc0.shape = {enc0.shape}")
        enc1 = self.encoder2(hidden_states_out[0])
        print(f"[DEBUG] enc1.shape = {enc1.shape}")
        enc2 = self.encoder3(hidden_states_out[1])
        print(f"[DEBUG] enc2.shape = {enc2.shape}")
        enc3 = self.encoder4(hidden_states_out[2])
        print(f"[DEBUG] enc3.shape = {enc3.shape}")
    
        dec4 = self.encoder10(hidden_states_out[4])
        print(f"[DEBUG] dec4.shape = {dec4.shape}")
    
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        print(f"[DEBUG] dec3.shape = {dec3.shape}")
        dec2 = self.decoder4(dec3, enc3)
        print(f"[DEBUG] dec2.shape = {dec2.shape}")
        dec1 = self.decoder3(dec2, enc2)
        print(f"[DEBUG] dec1.shape = {dec1.shape}")
        dec0 = self.decoder2(dec1, enc1)
        print(f"[DEBUG] dec0.shape = {dec0.shape}")
    
        out = self.decoder1(dec0, enc0)
        print(f"[DEBUG] out.shape = {out.shape}")
    
        logits = self.out(out)
        print(f"[DEBUG] logits.shape = {logits.shape}")

        return logits
    """