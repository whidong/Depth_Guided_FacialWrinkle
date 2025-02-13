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

from datasets.data_simmim import build_loader_simmim
from datasets.dataset import WrinkleDataset, get_texture_augmentations, get_pre_augmentations, WrappedDataset
from utils.train_utils import train_epoch, train_denoise_epoch, validate_epoch_pretrain, validate_epoch_denoise,save_model, save_pretraining_results, get_folder_name, collect_file_paths
from utils.train_utils import train_mask_epoch, validate_epoch_mask
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

def main_pretrain(model_type, run_id, mode, seed, batch_size = 23, epochs = 150):
    """
    한 번의 학습+검증 프로세스를 실행하고, 결과를 반환.
    run_id : 현재 실행 중인 반복 실험 번호
    total_runs : 총 반복 실험 횟수
    """
    # Random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(42) # Dataset suffle seed

    print(f"Starting experiment run_id={run_id}, seed={seed}")
    # Create model
    if model_type == "custom_unet":
        # in_channels by mode
        if mode == "RGB":
            in_ch = 3
            out_ch = 1
        elif mode == "RGBD":
            in_ch = 4
            out_ch = 1
        elif mode == "denoise":
            in_ch = 3
            out_ch = 3
        else:
            raise ValueError(f"Unknown mode: {mode}")
        model = create_model(model_type= model_type, 
                             in_channels = in_ch, 
                             out_channels = out_ch)
        model = nn.DataParallel(model).cuda()
    elif model_type == "custom_unetr":
        # in_channels by mode
        if mode == "RGB":
            in_ch = 3
            out_ch = 1
        elif mode == "RGBD":
            in_ch = 4
            out_ch = 1
        else:
            raise ValueError(f"Unknown mode: {mode}")
        model = create_model(model_type= model_type, 
                             img_size= (1024, 1024), 
                             in_channels = in_ch, 
                             out_channels = out_ch,
                             feature_size = 48,
                             use_checkpoint= True,
                             depth = (2, 2, 2, 2),
                             use_v2 = False)
        model = nn.DataParallel(model).cuda()
    elif model_type == "maked_swin":
        if mode == "masked":
            in_ch = 3
            out_ch = 3
        else:
            raise ValueError(f"Unknown mode: {mode}")
        model = create_model(model_type= model_type, 
                             img_size= (1024, 1024), 
                             in_channels = in_ch,
                             feature_size = 48,
                             use_checkpoint= True,
                             depth = (2, 2, 2, 2),
                             use_v2 = False)
        model = nn.DataParallel(model).cuda()


    # data path
    rgb_dir = "../data/images1024x1024"
    depth_dir = "../data/depth_masking"
    label_dir = "../data/weak_wrinkle_masks"

    # Get Ground Truth file name
    manual_mask_dir = "../data/manual_wrinkle_masks"
    manual_mask_files = []

    for dirpath, dirnames, filenames in os.walk(manual_mask_dir):
        for filename in filenames:
            if filename.endswith('.png'):
                manual_mask_files.append(filename)

    manual_mask_files = set(manual_mask_files)  # speed up by set 
    
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
      
    min_depth, max_depth = calculate_depth_min_max(depth_paths)
    print(f"Dataset : {len(data)} Mode : {mode} Model : {model_type} input : {in_ch} output : {out_ch}")
    print(f"Depth 이미지의 최소값: {min_depth}, 최대값: {max_depth}")

    # 데이터 분할
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_data, val_data = random_split(data, [train_size, val_size], generator=generator)
    print(f"Size of dataset train : {train_size} val : {val_size}")
    # RGB + D + Label
    train_rgb_paths, train_depth_paths, train_label_paths = zip(*train_data)
    val_rgb_paths, val_depth_paths, val_label_paths = zip(*val_data)

    if mode == "RGBD" :
      train_transform = get_pre_augmentations()
      val_transform = A.Compose([
      A.Normalize(mean=(0.485, 0.456, 0.406, 0), std=(0.229, 0.224, 0.225, 1)),
      ToTensorV2(transpose_mask=True)
      ])

    elif mode == "RGB":
      train_transform = get_texture_augmentations()
      val_transform = A.Compose([
          A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
          ToTensorV2(transpose_mask=True)
      ])
    
    elif mode == "denoise":
        # denoise 모드 전처리 정의
        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(transpose_mask=True),
        ])
        val_transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(transpose_mask=True),
        ])
    elif mode == "masked":
        pass

    if mode == "masked":
        train_loader = build_loader_simmim(data_path=train_rgb_paths, batch_size=batch_size, shuffle=True, num_workers=8, validation=False)
        val_loader = build_loader_simmim(data_path=val_rgb_paths, batch_size=batch_size, shuffle=False, num_workers=8, validation=True)
    else:
        # 데이터셋 생성
        train_dataset = WrinkleDataset(train_rgb_paths, train_depth_paths, label_paths = train_label_paths, transform=train_transform, min_depth=min_depth, max_depth=max_depth, mode = mode)
        val_dataset = WrinkleDataset(val_rgb_paths, val_depth_paths, label_paths = val_label_paths, transform=val_transform, min_depth=min_depth, max_depth=max_depth, mode = mode)

        # 데이터로더 생성
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    tensorboard_dir = "./tensorboard"
    model_pth_dir = "./checkpoint"
    if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
    if not os.path.exists(model_pth_dir):
            os.makedirs(model_pth_dir)

    # 손실함수, 옵티마이저, 스케줄러
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
    model.parameters(),   # 모델의 학습 가능한 파라미터
    lr=0.001,             # 학습률
    weight_decay=0.05,    # Weight Decay (L2 Regularization)
    betas=(0.9, 0.999)    # β1=0.9, β2=0.999
    )
    scheduler = CustomCosineAnnealingWarmRestarts(optimizer=optimizer, T_0=100, T_mult=2, eta_min=0, eta_max=0.001, decay_factor=0.9, start_epoch=0)
    epochs = epochs
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=f'./tensorboard/{model_type}_pretrain_{mode}_{run_id}_seed{seed}')
    best_val_loss_unet = float('inf')
    patience = 15
    patience_counter = 0
    

    for epoch in range(epochs):
        print(f"[Run {run_id+1}] Epoch {epoch + 1}/{epochs}")
        if mode == "denoise":
            train_loss = train_denoise_epoch(train_loader, model, criterion, optimizer, scaler)
            val_loss = validate_epoch_denoise(val_loader, model, criterion, epoch + 1, writer = writer)
        elif mode == "masked":
            train_loss = train_mask_epoch(train_loader, model, optimizer, scaler)
            val_loss = validate_epoch_mask(val_loader, model, epoch + 1, writer = writer)
        else:
            train_loss = train_epoch(train_loader, model, criterion, optimizer, scaler)
            val_loss = validate_epoch_pretrain(val_loader, model, criterion, epoch + 1, writer = writer)

        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Run {run_id+1}] Epoch {epoch+1}: LR={current_lr:.8f}, TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}")

        writer.add_scalar('Loss/Train', train_loss, epoch + 1)
        writer.add_scalar('Loss/Validation', val_loss, epoch + 1)
        writer.add_scalar('Learning Rate', current_lr, epoch + 1)
        if mode == "denoise":
            if val_loss < best_val_loss_unet:
                best_val_loss_unet = val_loss
                from utils.train_utils import save_model
                save_model(model, optimizer, scheduler, epoch + 1, best_val_loss_unet, f'./checkpoint/best_pretrain_{mode}_{run_id}_seed{seed}.pth')
                print(f"[Run {run_id+1}] Model saved based on Validation Loss: {val_loss:.6f}")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"[Run {run_id+1}] No improvement. Patience counter: {patience_counter}/{patience} Validation Loss: {val_loss:.6f}")
        elif mode == "masked":
            if val_loss < best_val_loss_unet:
                best_val_loss_unet = val_loss
                from utils.train_utils import save_model
                save_model(model, optimizer, scheduler, epoch + 1, best_val_loss_unet, f'./checkpoint/best_pretrain_{mode}_{run_id}_seed{seed}.pth')
                print(f"[Run {run_id+1}] Model saved based on Validation Loss: {val_loss:.6f}")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"[Run {run_id+1}] No improvement. Patience counter: {patience_counter}/{patience} Validation Loss: {val_loss:.6f}")
        else: 
            if val_loss < best_val_loss_unet:
                best_val_loss_unet = val_loss
                from utils.train_utils import save_model
                save_model(model, optimizer, scheduler, epoch + 1, best_val_loss_unet, f'./checkpoint/best_pretrain_{mode}_{run_id}_seed{seed}.pth')
                print(f"[Run {run_id+1}] Model saved based on Validation Loss: {val_loss:.6f}")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"[Run {run_id+1}] No improvement. Patience counter: {patience_counter}/{patience} Validation Loss: {val_loss:.6f} Acc: {val_acc:.6f}")

    writer.close()
    print(f"[Run {run_id+1}] Finetuning Completed.")
    del model
    torch.cuda.empty_cache()
    return best_val_loss_unet

def main():
    epochs = 300
    run_seeds = [42, 42, 42, 42, 42, 42] # pretrain model seed
    mode = ["RGB","RGBD","denoise","RGB", "RGBD","masked"] # pretrain model mode
    results = [] # pretrain result
    batch_size = [36, 36, 36, 36, 36, 36]
    model_type = ["custom_unet","custom_unet", "custom_unet","custom_unetr","custom_unetr","maked_swin"]

    for run_id, seed in enumerate(run_seeds):
        val_loss = main_pretrain(model_type[run_id], run_id, mode[run_id], seed, batch_size[run_id], epochs)
        results.append((val_loss))

    print("=== Final Results of Runs ===")
    for i, loss in enumerate(results):
        print(f"Run {i+1}: {mode[i]} Best Val Loss = {loss:.4f}")

if __name__ == "__main__":
    main()