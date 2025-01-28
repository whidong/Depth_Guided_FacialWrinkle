import os
import re
import torch
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import calculate_metrics_gpu

def train_epoch(loader, model, criterion, optimizer, scaler):
    model.train()
    epoch_loss = 0
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()

        # Forward
        with autocast(device_type='cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # Backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def validate_epoch(loader, model, criterion, epoch, scaler, writer=None):
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(loader, desc="Validation")):
            inputs, labels = inputs.cuda(), labels.cuda()

            # Forward
            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            labels = labels.squeeze(1)
            all_preds.append(preds)
            all_labels.append(labels)

            # Save some predictions for visualization (차원 복구)
            if batch_idx < 5 and writer is not None:
                save_pretraining_results(inputs, outputs, labels, epoch, batch_idx, writer)

        # Calculate metrics
    all_preds = torch.cat(all_preds, dim=0).cuda()
    all_labels = torch.cat(all_labels, dim=0).cuda()
    jsi, f1, acc, precision, recall = calculate_metrics_gpu(all_labels, all_preds)
    
    # Log metrics to TensorBoard
    if writer is not None:
        writer.add_scalar('Metrics/JSI', jsi, epoch)
        writer.add_scalar('Metrics/F1-Score', f1, epoch)
        writer.add_scalar('Metrics/Accuracy', acc, epoch)
        writer.add_scalar('Metrics/Precision', precision, epoch)
        writer.add_scalar('Metrics/Recall', recall, epoch)

    # Return loss and metrics
    return epoch_loss / len(loader), jsi, f1

def validate_epoch_pretrain(loader, model, criterion, epoch, scaler, writer=None):
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(loader, desc="Validation")):
            inputs, labels = input.cuda(), labels.cuda()
            
            # Forward
            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            labels = labels.squeeze(1)
            all_preds.append(preds)
            all_labels.append(labels)

            # Save some predictions for visualization (차원 복구)
            if batch_idx < 5 and writer is not None:
                save_pretraining_results(inputs, outputs, labels, epoch, batch_idx, writer)

        # Calculate metrics
    all_preds = torch.cat(all_preds, dim=0).cuda()
    all_labels = torch.cat(all_labels, dim=0).cuda()
    jsi, f1, acc, precision, recall = calculate_metrics_gpu(all_labels, all_preds)

    # Log metrics to TensorBoard
    if writer is not None:
        writer.add_scalar('Metrics/JSI', jsi, epoch)
        writer.add_scalar('Metrics/F1-Score', f1, epoch)
        writer.add_scalar('Metrics/Accuracy', acc, epoch)
        writer.add_scalar('Metrics/Precision', precision, epoch)
        writer.add_scalar('Metrics/Recall', recall, epoch)

    return epoch_loss / len(loader), acc, f1






def save_pretraining_results(inputs, outputs, labels, epoch, batch_idx, writer=None):
    # 배치에서 첫 번째 이미지 사용
    input_image = inputs[0].cpu().detach()
    output_image = outputs[0].cpu().detach()
    label_image = labels[0].cpu().detach()
    
    # 입력 이미지의 RGB 채널만 사용
    input_image_rgb = input_image[:3]  # [3, H, W]
    
    # 시각화를 위해 [0, 1] 범위로 스케일링
    input_image_rgb = (input_image_rgb - input_image_rgb.min()) / (input_image_rgb.max() - input_image_rgb.min())
    
    # 예측 마스크: torch.argmax을 사용하여 클래스 인덱스 추출
    pred_mask = torch.argmax(output_image, dim=0).float()  # [H, W]
    pred_mask = pred_mask.unsqueeze(0)  # [1, H, W]
    # 필요에 따라 스케일링 (이미 이진화된 경우 생략 가능)
    pred_mask = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min())
    
    # 실제 마스크: 채널 차원 제거 후 채널 차원 추가
    true_mask = label_image.squeeze(0).float()  # [H, W]
    true_mask = true_mask.unsqueeze(0)  # [1, H, W]
    # 필요에 따라 스케일링 (이미 이진화된 경우 생략 가능)
    true_mask = (true_mask - true_mask.min()) / (true_mask.max() - true_mask.min())
    
    # TensorBoard에 이미지 추가
    writer.add_image(f'Validation/Input_Epoch_{epoch}_Batch_{batch_idx}', input_image_rgb, epoch)
    writer.add_image(f'Validation/Predicted_Epoch_{epoch}_Batch_{batch_idx}', pred_mask, epoch)
    writer.add_image(f'Validation/GroundTruth_Epoch_{epoch}_Batch_{batch_idx}', true_mask, epoch)

def save_model(model, optimizer, scheduler, epoch, best_val_loss, filepath='best_model_checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss
    }, filepath)
    print(f"Model saved to {filepath}")

# 모든 파일 경로 수집 함수
def collect_file_paths(root_dir, manual_mask_files, start_folder=0, end_folder=49000):
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 현재 폴더의 이름 추출
        current_folder = os.path.basename(dirpath)
        # 폴더 이름이 숫자로만 이루어져 있는지 확인
        if current_folder.isdigit():
            folder_number = int(current_folder)
            # 폴더 번호가 범위 내에 있는지 확인
            if start_folder <= folder_number <= end_folder:
                for filename in filenames:
                    if filename.endswith('.png') and filename not in manual_mask_files:
                        # 파일 이름에서 숫자 부분 추출
                        file_base_name = filename[:-4]  # '.png' 제거
                        # 숫자로만 이루어진 파일 이름인지 확인
                        if re.fullmatch(r'\d+', file_base_name):
                            file_paths.append(os.path.join(dirpath, filename))
    return sorted(file_paths)

# 매칭된 파일 경로 리스트 생성
def get_folder_name(file_name):
    file_base_name = file_name[:-4]  # '.png' 제거
    # 파일 이름에서 숫자 부분 추출
    number_str = ''.join(filter(str.isdigit, file_base_name))
    if number_str == '':
        raise ValueError(f"파일 이름에서 숫자를 추출할 수 없습니다: {file_name}")
    number = int(number_str)
    folder_number = (number // 1000) * 1000
    folder_name = f"{folder_number:05d}"
    return folder_name