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

from datasets.dataset import WrinkleDataset, get_texture_augmentations, get_pre_augmentations, WrappedDataset
from utils.train_utils import train_epoch, validate_epoch_pretrain, save_model, save_pretraining_results, get_folder_name, collect_file_paths
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

def main_pretrain(run_id, mode, seed, batch_size = 8):
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

    # in_channels by mode
    if mode == "RGB":
        in_ch = 3
    elif mode == "RGBD":
        in_ch = 4
    else:
        raise ValueError(f"Unknown mode: {mode}")

    model = create_model(model_type="custom_unet", in_channels=in_ch, out_channels=1)
    model = nn.DataParallel(model).cuda()

    # 경로 설정
    rgb_dir = "../data/images1024x1024"
    depth_dir = "../data/depth_masking"
    label_dir = "../data/weak_wrinkle_masks"

    # Ground Truth 주름 마스크 파일 이름 수집
    manual_mask_dir = "../data/manual_wrinkle_masks"
    manual_mask_files = []

    for dirpath, dirnames, filenames in os.walk(manual_mask_dir):
        for filename in filenames:
            if filename.endswith('.png'):
                manual_mask_files.append(filename)

    manual_mask_files = set(manual_mask_files)  # 집합으로 변환하여 검색 속도 향상
    
    # RGB, Depth, Label 파일 경로 수집
    rgb_paths = collect_file_paths(rgb_dir, manual_mask_files, start_folder=0, end_folder=49000)
    depth_paths = collect_file_paths(depth_dir, manual_mask_files, start_folder=0, end_folder=49000)
    label_paths = collect_file_paths(label_dir, manual_mask_files, start_folder=0, end_folder=49000)

    # 파일 이름만 추출하여 리스트 생성
    rgb_files = [os.path.basename(path) for path in rgb_paths]
    depth_files = [os.path.basename(path) for path in depth_paths]
    label_files = [os.path.basename(path) for path in label_paths]

    # 파일 이름이 동일한 것들만 매칭
    common_files = set(rgb_files) & set(depth_files) & set(label_files)
    common_files = sorted(list(common_files))

    # 매칭된 파일 경로 리스트 생성
    rgb_paths = [os.path.join(rgb_dir, get_folder_name(f), f) for f in common_files]
    depth_paths = [os.path.join(depth_dir, get_folder_name(f), f) for f in common_files]
    label_paths = [os.path.join(label_dir, get_folder_name(f), f) for f in common_files]

    # 파일 경로 존재 여부 확인 및 필터링
    rgb_paths = [path for path in rgb_paths if os.path.exists(path)]
    depth_paths = [path for path in depth_paths if os.path.exists(path)]
    label_paths = [path for path in label_paths if os.path.exists(path)]
    
    # 데이터셋 분할을 위한 파일 경로 리스트 생성
    data = list(zip(rgb_paths, depth_paths, label_paths))
    random.shuffle(data)
      
    min_depth, max_depth = calculate_depth_min_max(depth_paths)
    print(f"Depth 이미지의 최소값: {min_depth}, 최대값: {max_depth}")

    # 데이터 분할
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]

    # RGB + D + Label
    train_rgb_paths, train_depth_paths, train_label_paths = zip(*train_data)
    val_rgb_paths, val_depth_paths, val_label_paths = zip(*val_data)

    if mode == "RGBT" :
      train_transform = get_pre_augmentations()
      val_transform = A.Compose([
      A.Normalize(mean=(0.485, 0.456, 0.406, 0.0), std=(0.229, 0.224, 0.225, 1.0)),
      ToTensorV2()
      ])

    elif mode == "RGB":
      train_transform = get_texture_augmentations()
      val_transform = A.Compose([
          A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
          ToTensorV2(transpose_mask=True)
      ])

    # 데이터셋 생성
    train_dataset = WrinkleDataset(train_rgb_paths, train_depth_paths, train_label_paths, transform=train_transform, min_depth=min_depth, max_depth=max_depth, mode = mode)
    val_dataset = WrinkleDataset(val_rgb_paths, val_depth_paths, val_label_paths, transform=val_transform, min_depth=min_depth, max_depth=max_depth, mode = mode)

    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 손실함수, 옵티마이저, 스케줄러
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
    model.parameters(),   # 모델의 학습 가능한 파라미터
    lr=0.001,             # 학습률
    weight_decay=0.05,    # Weight Decay (L2 Regularization)
    betas=(0.9, 0.999)    # β1=0.9, β2=0.999
    )
    scheduler = CustomCosineAnnealingWarmRestarts(optimizer=optimizer, T_0=100, T_mult=2, eta_min=0, eta_max=0.001, decay_factor=0.9, start_epoch=0)

    epochs = 300
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=f'unet_runs/unet_pretrain_{mode}_seed{seed}')
    best_val_loss_unet = float('inf')
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        print(f"[Run {run_id+1}] Epoch {epoch + 1}/{epochs}")
        train_loss = train_epoch(train_loader, model, criterion, optimizer, scaler)
        val_loss, val_acc, val_f1 = validate_epoch_pretrain(val_loader, model, criterion, epoch + 1, scaler, writer = writer)

        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Run {run_id+1}] Epoch {epoch+1}: LR={current_lr:.8f}, TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}")

        writer.add_scalar('Loss/Train', train_loss, epoch + 1)
        writer.add_scalar('Loss/Validation', val_loss, epoch + 1)
        writer.add_scalar('Learning Rate', current_lr, epoch + 1)

        if val_loss < best_val_loss_unet:
            best_val_loss_unet = val_loss
            from utils.train_utils import save_model
            save_model(model, optimizer, scheduler, epoch + 1, best_val_loss_unet, f'./no_RDT/best_pretrain_{mode}_seed{seed}.pth')
            print(f"[Run {run_id+1}] Model saved based on Validation Loss: {val_loss:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"[Run {run_id+1}] No improvement. Patience counter: {patience_counter}/{patience}")

    writer.close()
    print(f"[Run {run_id+1}] Finetuning Completed.")
    del model
    torch.cuda.empty_cache()
    return best_val_loss_unet

def main():
    run_seeds = [42, 42]
    mode = ["RGB", "RGBD"]
    results = []
    batch_size = 8

    for run_id, seed in enumerate(run_seeds):
        val_loss = main_pretrain(run_id, mode[run_id], seed, batch_size)
        results.append((val_loss))

    print("=== Final Results of Runs ===")
    for i, loss in enumerate(results):
        print(f"Run {i+1}: {mode[i]} Best Val Loss = {loss:.4f}")

if __name__ == "__main__":
    main()