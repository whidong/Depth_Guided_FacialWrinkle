import sys
import random
sys.path.append("../")  # 프로젝트 루트를 경로에 추가
from model import create_model
import re 
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from datasets.dataset import WrinkleDataset, get_train_augmentations, WrappedDataset
from utils.train_utils import train_epoch, validate_epoch, save_model, save_pretraining_results
from utils.metrics import calculate_metrics_gpu, calculate_depth_min_max
from utils.custom_scheduler import CustomCosineAnnealingWarmRestarts
from loss.dice_loss import soft_dice_loss

import numpy as np
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim

def main_experiment(run_id, seed, mode, model_type, batch_size = 10):
    """
    한 번의 학습+검증 프로세스를 실행하고, 결과를 반환.
    run_id : 현재 실행 중인 반복 실험 번호
    total_runs : 총 반복 실험 횟수
    """
    # 랜덤 시드
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(42)

    print(f"Starting experiment run_id={run_id}, seed={seed}")
    
    if model_type == "custom_unet":
        if mode == "RGB": #imagenet or denoise
            in_ch = 3
            out_ch = 2
        elif mode == "RGBT":
            in_ch = 4
            out_ch = 2
        elif mode == "RGBDT":
            in_ch = 5
            out_ch = 2
        elif mode == "denoise":
            in_ch = 3
            out_ch = 2
        else:
            raise ValueError(f"Unknown mode: {mode}")

         unet_fine_model = create_model(
            model_type="custom_unet",
            in_channels=in_ch,   # 입력 채널 수: RGB(3) + Depth(1) + Weak Texture Map(1)
            out_channels=out_ch   # 출력 채널 수: Wrinkle(1) + Background(1)
        )
        unet_fine_model = nn.DataParallel(unet_fine_model).cuda()
    elif model_type == "custom_unetr":
        if mode == "RGB": #imagenet or mask
            in_ch = 3
            out_ch = 2
        elif mode == "RGBT":
            in_ch = 4
            out_ch = 2
        elif mode == "RGBDT":
            in_ch = 5
            out_ch = 2
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        unetr_fine_model = create(
            model_type="custom_unetr",
            img_size = (1024, 1024),
            in_channels = in_ch,
            out_channels = out_ch,
            feature_size = 48,
            use_checkpoint = True,
            depth = (2, 2, 2, 2),
            use_v2 = False)
        unetr_fine_model = nn.DataParallel(unetr_fine_model).cuda()
            


    # 경로 설정
    rgb_dir = "../data/finetuning/masked_face_images"
    depth_dir = "../data/finetuning/depth_masking"
    weak_texture_dir = "../data/finetuning/weak_wrinkle_mask"
    label_dir = "../data/finetuning/manual_wrinkle_masks"

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
    print(f"Depth 이미지의 최소값: {min_depth}, 최대값: {max_depth}")

    dataset = WrinkleDataset(rgb_paths, depth_paths, weak_texture_paths, label_paths,
                             transform=None, min_depth=min_depth, max_depth=max_depth, mode=mode, task = "finetune")
    
    # 데이터 분할
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_subset, val_subset, test_subset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    train_transform = get_train_augmentations()
    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406, 0, 0), std=(0.229, 0.224, 0.225, 1, 1)),
        ToTensorV2(transpose_mask=True)
    ])

    train_dataset = WrappedDataset(train_subset, transform=train_transform)
    val_dataset = WrappedDataset(val_subset, transform=val_transform)
    test_dataset = WrappedDataset(test_subset, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 손실함수, 옵티마이저, 스케줄러
    criterion = lambda outputs, labels: soft_dice_loss(outputs, labels, smooth=1e-6,to_onehot_y=True, apply_softmax=True,include_background=True,n_classes=2)
    optimizer = optim.AdamW(unet_fine_model.parameters(), lr=0.0001, weight_decay=0.05, betas=(0.9, 0.999))
    scheduler = CustomCosineAnnealingWarmRestarts(optimizer=optimizer, T_0=50, T_mult=2, eta_min=0, eta_max=0.0001, decay_factor=0.9, start_epoch=0)

    tensorboard_dir = "./tensorboard/finetuning"
    model_pth_dir = "./checkpoint/finetuning"
    if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
    if not os.path.exists(model_pth_dir):
            os.makedirs(model_pth_dir)
        
    epochs = 150
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=f'./tensorboard/finetuning/{model_type}_fintuning_{mode}_{run_id}_seed{seed}')
    best_val_loss_unet = float('inf')
    best_val_jsi = 0.0
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        print(f"[Run {run_id+1}] Epoch {epoch + 1}/{epochs}")
        train_loss = train_epoch(train_loader, unet_fine_model, criterion, optimizer, scaler)
        val_loss, val_jsi, val_f1 = validate_epoch(val_loader, unet_fine_model, criterion, epoch + 1, writer = writer)

        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Run {run_id+1}] Epoch {epoch+1}: LR={current_lr:.8f}, TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}")

        writer.add_scalar('Loss/Train', train_loss, epoch + 1)
        writer.add_scalar('Loss/Validation', val_loss, epoch + 1)
        writer.add_scalar('Learning Rate', current_lr, epoch + 1)

        if val_jsi > best_val_jsi and val_loss < best_val_loss_unet:
            best_val_loss_unet = val_loss
            best_val_jsi = val_jsi
            from utils.train_utils import save_model
            save_model(unet_fine_model, optimizer, scheduler, epoch + 1, best_val_loss_unet, f'./checkpoint/finetuning/best_{model_type}_finetuning_{mode}_{run_id}_seed{seed}.pth')
            print(f"[Run {run_id+1}] Model saved based on Validation JSI: {val_jsi:.4f} and Loss: {val_loss:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"[Run {run_id+1}] No improvement. Patience counter: {patience_counter}/{patience}")

    writer.close()
    print(f"[Run {run_id+1}] Finetuning Completed.")
    del unet_fine_model
    torch.cuda.empty_cache()
    return best_val_loss_unet, best_val_jsi

def main():
    run_seeds = [42,2025,2024]
    mode = "RGB"
    batch_size = 10
    model_type = "custom_unet"
    results = []
    for run_id, seed in enumerate(run_seeds):
        val_loss, val_jsi = main_experiment(run_id, seed, mode, model_type,batch_size)
        results.append((val_loss, val_jsi))

    print("=== Final Results of 3 Runs ===")
    for i, (loss, jsi) in enumerate(results):
        print(f"Run {i+1}: Best Val Loss = {loss:.4f}, Best JSI = {jsi:.4f}")
    
    # 평균, 표준편차 계산
    losses = [r[0] for r in results]
    jsis = [r[1] for r in results]
    print(f"Avg. Loss = {np.mean(losses):.4f} ± {np.std(losses):.4f}")
    print(f"Avg. JSI  = {np.mean(jsis):.4f} ± {np.std(jsis):.4f}")

if __name__ == "__main__":
    main()
