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
from utils.train_utils import train_epoch, validate_epoch, save_model, save_pretraining_results, load_pretrain
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

def main_experiment(run_id, seed, mode, model_type, epoch, batch_size = 10, pretrain_path=None):
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
        in_ch = {"RGB": 3, "RGBT": 4, "RGBDT": 5, "denoise": 3}.get(mode, None)
        out_ch = 2
        if in_ch is None:
            raise ValueError(f"Unknown mode: {mode}")

        model = create_model(
            model_type= model_type,
            in_channels=in_ch,   # 입력 채널 수 RGB(3) + Depth(1) + Weak Texture Map(1)
            out_channels= out_ch  # 출력 채널 수 Wrinkle(1) + Background(1)
        )
    elif model_type == "imagenet_unet":
        in_ch = 3
        out_ch = 2
        model = create_model(
            model_type = model_type,
            in_channels = in_ch,
            out_channels = out_ch
        )
        
    elif model_type == "custom_unetr":
        params = {
            "RGB": (3, 2, 48, (2, 2, 2, 2), False),
            "RGBT": (4, 2, 48, (2, 2, 2, 2), False),
            "RGBDT": (5, 2, 48, (2, 2, 2, 2), False),
            "mask": (3, 2, 48, (2, 2, 2, 2), False),
            "image": (3, 2, 96, (2, 2, 6, 2), True),
        }

        if mode not in params:
            raise ValueError(f"Unknown mode: {mode}")

        in_ch, out_ch, feature, depths, v2 = params[mode]
            
        model = create_model(
            img_size= (1024, 1024), 
            model_type = model_type,
            in_channels = in_ch,
            out_channels = out_ch,
            feature_size = feature,
            use_v2 = v2,
            depth = depths,
            use_checkpoint= True,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    model = nn.DataParallel(model).cuda()

    if pretrain_path:
        model = load_pretrain(model, pretrain_path)
    
    print("Model initialized and ready for training!")
    
    
    # 경로 설정
    rgb_dir = "../data/finetuning/masked_face_images"
    depth_dir = "../data/finetuning/depth_masking_any_metric_full"
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
    dataset = WrinkleDataset(rgb_paths, depth_paths, weak_texture_paths, label_paths,
                             transform=None, min_depth=min_depth, max_depth=max_depth, mode=mode, task = "finetune")
    
    print(f"Dataset : {len(dataset)} Mode : {mode} Model : {model_type} input : {in_ch} output : {out_ch}")
    print(f"Depth 이미지의 최소값: {min_depth}, 최대값: {max_depth}")
    
    # 데이터 분할
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    print(f"Size of dataset train : {train_size} val : {val_size} test : {test_size}")
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
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05, betas=(0.9, 0.999))
    scheduler = CustomCosineAnnealingWarmRestarts(optimizer=optimizer, T_0= epoch // 3, T_mult=2, eta_min=0, eta_max=0.0001, decay_factor=0.9, start_epoch=0)
    tensorboard_dir = "./tensorboard/fine_tuning"
    model_pth_dir = f"./checkpoint/fine_tuning/{model_type}_{mode}"

    if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
    if not os.path.exists(model_pth_dir):
            os.makedirs(model_pth_dir)
    epochs = epoch
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=f'./tensorboard/fine_tuning/{model_type}_{mode}_{run_id}_seed{seed}')
    best_val_loss_unet = float('inf')
    best_val_jsi = 0.0
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        print(f"[Run {run_id+1}] Epoch {epoch + 1}/{epochs}")
        train_loss = train_epoch(train_loader, model, criterion, optimizer, scaler)
        val_loss, val_jsi, val_f1 = validate_epoch(val_loader, model, criterion, epoch + 1, writer = writer)

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
            save_model(model, optimizer, scheduler, epoch + 1, best_val_loss_unet, f'./checkpoint/fine_tuning/{model_type}_{mode}/best_{model_type}_{mode}_{run_id}_seed{seed}.pth')
            print(f"[Run {run_id+1}] Model saved based on Validation JSI: {val_jsi:.4f} and Loss: {val_loss:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"[Run {run_id+1}] No improvement. Patience counter: {patience_counter}/{patience}")

    writer.close()
    print(f"[Run {run_id+1}] Finetuning Completed.")
    del model
    torch.cuda.empty_cache()
    return best_val_loss_unet, best_val_jsi

def main():
    run_seeds = [42,2025,2024]
    experiments = {
        "custom_unet": {
            "modes": {
                "RGB": "./checkpoint/pretrain/best_pretrain_unet_RGB_seed42.pth",
                "RGBT": "./checkpoint/pretrain/best_pretrain_unet_RGB_seed42.pth",
                "RGBDT": "./checkpoint/pretrain/best_pretrain_unet_RGBD_seed42.pth"
            },
            "epochs": 3
        },
        "imagenet_unet": {
            "modes": {
                "denoise": "./checkpoint/pretrain/best_pretrain_unet_denoise_seed42.pth",
                "RGB": None
            },
            "epochs": 3
        },
        "custom_unetr": {
            "modes": {
                "RGB": "./checkpoint/pretrain/best_pretrain_swin_unetr_RGB_seed42.pth",
                "RGBT": "./checkpoint/pretrain/best_pretrain_swin_unetr_RGB_seed42.pth",
                "RGBDT": "./checkpoint/pretrain/best_pretrain_swin_unetr_RGBD_seed42.pth",
                "image": "./checkpoint/pretrain/best_pretrain_swin_unetr_image_seed42.pth",
                "mask" : "./checkpoint/pretrain/best_pretrain_swin_unetr_simmim_r_seed42.pth"
            },
            "epochs": 3
        }
    }
    
    batch_size = 4
    results = []
    
    for model_type, settings in experiments.items():
        modes_dict  = settings["modes"]
        epochs = settings["epochs"] 

        print(f"\n=== Running experiments for model: {model_type} (epochs={epochs}) ===")
        model_results = []  # 특정 모델의 결과 저장

        for mode, pretrain_path in modes_dict.items():
            for run_id, seed in enumerate(run_seeds):
                print(f"\n▶ Running {model_type} with mode {mode}, seed {seed}, epochs {epochs} (run_id={run_id})")
                print(f"   Using pretrain path: {pretrain_path}")
                
                val_loss, val_jsi = main_experiment(run_id, seed, mode, model_type, epochs, batch_size, pretrain_path)
                
                # 결과 저장
                model_results.append((model_type, mode, seed, val_loss, val_jsi))
                results.append((model_type, mode, seed, val_loss, val_jsi))

        # 특정 모델 타입의 결과 요약 출력
        losses = [r[3] for r in model_results]
        jsis = [r[4] for r in model_results]
        print(f"\n=== {model_type} Summary (epochs={epochs}) ===")
        print(f"Avg. Loss = {np.mean(losses):.4f} ± {np.std(losses):.4f}")
        print(f"Avg. JSI  = {np.mean(jsis):.4f} ± {np.std(jsis):.4f}")

    # 전체 실험 결과 출력
    print("\n=== Final Results of All Runs ===")
    for model_type, mode, seed, loss, jsi in results:
        print(f"[{model_type}] Mode: {mode}, Seed: {seed} -> Loss: {loss:.4f}, JSI: {jsi:.4f}")

    # 전체 평균 및 표준편차 계산
    all_losses = [r[3] for r in results]
    all_jsis = [r[4] for r in results]
    print(f"\nFinal Avg. Loss = {np.mean(all_losses):.4f} ± {np.std(all_losses):.4f}")
    print(f"Final Avg. JSI  = {np.mean(all_jsis):.4f} ± {np.std(all_jsis):.4f}")

if __name__ == "__main__":
    main()
