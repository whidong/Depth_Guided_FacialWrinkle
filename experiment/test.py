import sys
import random
sys.path.append("../")  # 프로젝트 루트를 경로에 추가
from model import create_model

import os
import glob
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from datasets.dataset import WrinkleDataset, get_train_augmentations, WrappedDataset
from utils.metrics import evaluate_model_gpu, calculate_depth_min_max

def main_test(mode, ckpt_path, seed=42, batch_size=8):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(42)

    # in_channels by mode
    if mode == "RGB":
        in_ch = 3
    elif mode == "RGBT":
        in_ch = 4
    elif mode == "RGBDT":
        in_ch = 5
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 모델
    model = create_model(model_type="custom_unet", in_channels=in_ch, out_channels=2)
    model = nn.DataParallel(model).cuda()

    rgb_dir = "/home/donghwi/F_wrinkle_model_project/data/finetuning/masked_face_images"
    depth_dir = "/home/donghwi/F_wrinkle_model_project/data/finetuning/depth_masking"
    weak_texture_dir = "/home/donghwi/F_wrinkle_model_project/data/finetuning/weak_wrinkle_mask"
    label_dir = "/home/donghwi/F_wrinkle_model_project/data/finetuning/manual_wrinkle_masks"

    # 파일 리스트
    rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    depth_paths = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    weak_texture_paths = sorted(glob.glob(os.path.join(weak_texture_dir, "*.png")))
    label_paths = sorted(glob.glob(os.path.join(label_dir, "*.png")))

    # 파일명 정렬
    rgb_paths = sorted(rgb_paths, key=lambda x: os.path.basename(x))
    depth_paths = sorted(depth_paths, key=lambda x: os.path.basename(x))
    weak_texture_paths = sorted(weak_texture_paths, key=lambda x: os.path.basename(x))
    label_paths = sorted(label_paths, key=lambda x: os.path.basename(x))

    min_depth, max_depth = calculate_depth_min_max(depth_paths)

    dataset = WrinkleDataset(rgb_paths, depth_paths, weak_texture_paths, label_paths,
                             transform=None, min_depth=min_depth, max_depth=max_depth, mode=mode)
    # 데이터 분할
    total_len = len(dataset)
    train_size = int(0.8 * total_len)
    val_size   = int(0.1 * total_len)
    test_size = total_len - train_size - val_size
    train_subset, val_subset, test_subset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406, 0.0, 0.0), std=(0.229, 0.224, 0.225, 1.0, 1.0)),
        ToTensorV2(transpose_mask=True)
    ])

    test_dataset = WrappedDataset(test_subset, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # checkpoint 로드
    checkpoint = torch.load(ckpt_path, map_location='cuda')
    if 'model_state_dict' in checkpoint:
        pretrained_dict = checkpoint['model_state_dict']
    else:
        pretrained_dict = checkpoint

    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    model.load_state_dict(filtered_dict, strict=False)

    # evaluate
    jsi, f1, acc, prec, rec = evaluate_model_gpu(test_loader, model)
    print(f"[Mode={mode}, seed={seed}] ckpt={ckpt_path}")
    print(f"  => JSI={jsi:.5f}, F1={f1:.5f}, Acc={acc:.5f}, Prec={prec:.5f}, Rec={rec:.5f}")
    return jsi, f1, acc, prec, rec




def main():
    # 3개 모드
    mode_list  = ["RGB", "RGBT", "RGBDT"]

    # 예시로 아래와 같이 3개씩
    #   RGB -> (ckpt_rgb_1, ckpt_rgb_2, ckpt_rgb_3)
    #   RGBD-> (ckpt_rgbd_1, ckpt_rgbd_2, ckpt_rgbd_3)
    #   RGBDT-> ...
    all_checkpoints = {
        "RGB": [
            "./no_RDT/best_unet_finetuning_RGB_0_seed42.pth",
            "./no_RDT/best_unet_finetuning_RGB_1_seed2025.pth",
            "./no_RDT/best_unet_finetuning_RGB_2_seed2024.pth"
        ],
        "RGBT": [
            "./no_RDT/best_unet_finetuning_RGBT_no1.pth",
            "./no_RDT/best_unet_finetuning_RGBT_0_seed2025.pth",
            "./no_RDT/best_unet_finetuning_RGBT_0_seed2024.pth"
        ],
        "RGBDT": [
            "./no_RDT/best_unet_finetuning_RGBDT_no4.pth",
            "./no_RDT/best_unet_finetuning_RGBDT_no5.pth",
            "./no_RDT/best_unet_finetuning_RGBDT_no6.pth"
        ]
    }

    batch_size = 10
    seeds = [42, 2025, 2024]

    # 모드별로 결과 저장
    for mode in mode_list:
        ckpts  = all_checkpoints[mode]  # 3개
        results = []

        for ckpt_path, seed in zip(ckpts, seeds):
            jsi, f1, acc, prec, rec = main_test(mode, ckpt_path, batch_size = batch_size, seed = seed)
            results.append((jsi, f1, acc, prec, rec))
        
        results_array = np.array(results)
        means = results_array.mean(axis=0)
        stds  = results_array.std(axis=0)

        metric_names = ["JSI", "F1", "Acc", "Prec", "Recall"]
        print(f"\n==== Mode={mode} Final Results across seeds=[42,2025,2024] ====")
        for i, mname in enumerate(metric_names):
            print(f"{mname}: {means[i]:.5f} ± {stds[i]:.5f}")

if __name__ == "__main__":
    main()