import os
import re
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import calculate_metrics_gpu, add_noise
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim

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

def train_denoise_epoch(loader, model, criterion, optimizer, scaler):
    model.train()
    epoch_loss = 0
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs = inputs.cuda()
        optimizer.zero_grad()
        # Forward
        with autocast(device_type='cuda'):
            # noise
            x_noise, noise = add_noise(inputs, noise_type="scaled", noise_std = 0.22)
            outputs = model(x_noise)
            loss = criterion(outputs, noise)

        # Backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def train_mask_epoch(loader, model, optimizer, scaler):
    model.train()
    epoch_loss = 0
    for inputs, masks in tqdm(loader, desc="Training"):
        inputs, masks = inputs.cuda(), masks.cuda()

        optimizer.zero_grad()

        # Forward
        with autocast(device_type='cuda'):
            loss, result = model(inputs, masks)
        # print(loss, type(loss))
        # Backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def validate_epoch(loader, model, criterion, epoch, writer=None):
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

            torch.cuda.empty_cache()

        # Calculate metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    jsi, f1, acc, precision, recall = calculate_metrics_gpu(all_labels.cuda(), all_preds.cuda())
    
    # Log metrics to TensorBoard
    if writer is not None:
        writer.add_scalar('Metrics/JSI', jsi, epoch)
        writer.add_scalar('Metrics/F1-Score', f1, epoch)
        writer.add_scalar('Metrics/Accuracy', acc, epoch)
        writer.add_scalar('Metrics/Precision', precision, epoch)
        writer.add_scalar('Metrics/Recall', recall, epoch)

    # Return loss and metrics
    return epoch_loss / len(loader), jsi, f1

def validate_epoch_pretrain(loader, model, criterion, epoch, writer=None):
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
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

            # Save some predictions for visualization (차원 복구)
            if batch_idx < 5 and writer is not None:
                save_pretraining_results(inputs, outputs, labels, epoch, batch_idx, writer)

        # Calculate metrics
    #all_preds = torch.cat(all_preds, dim=0)
    #all_labels = torch.cat(all_labels, dim=0)
    #jsi, f1, acc, precision, recall = calculate_metrics_gpu(all_labels, all_preds)

    # Log metrics to TensorBoard
    #if writer is not None:
    #    writer.add_scalar('Metrics/JSI', jsi, epoch)
    #    writer.add_scalar('Metrics/F1-Score', f1, epoch)
    #    writer.add_scalar('Metrics/Accuracy', acc, epoch)
    #    writer.add_scalar('Metrics/Precision', precision, epoch)
    #    writer.add_scalar('Metrics/Recall', recall, epoch)

    return epoch_loss / len(loader)

def validate_epoch_denoise(loader, model, criterion, epoch, writer=None):
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(loader, desc="Validation")):
            inputs = inputs.cuda()
            
            # Forward
            with autocast(device_type='cuda'):
                # noise
                x_noise, noise = add_noise(inputs, noise_type="scaled", noise_std = 0.22)
                outputs = model(x_noise)
                loss = criterion(outputs, noise)
            epoch_loss += loss.item()

            preds = x_noise - outputs
            labels = labels
            #all_preds.append(preds.cpu())
            #all_labels.append(inputs.cpu())

            # Save some predictions for visualization (차원 복구)
            if batch_idx < 5 and writer is not None:
                save_pretraining_denoise_result(x_noise, preds, noise, outputs, epoch, batch_idx, writer)

    return epoch_loss / len(loader)

def validate_epoch_mask(loader, model, epoch, writer=None):
    model.eval()
    epoch_loss = 0
    #all_preds = []
    #all_labels = []

    with torch.no_grad():
        for batch_idx, (inputs, masks) in enumerate(tqdm(loader, desc="Validation")):
            inputs, masks = inputs.cuda(), masks.cuda()

            with autocast(device_type='cuda'):
                # Forward
                loss, outputs = model(inputs, masks)
            epoch_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            #all_preds.append(preds)
            #all_labels.append(masks)

            # Save some predictions for visualization (차원 복구)
            if batch_idx < 5 and writer is not None:
                save_pretraining_mask_results(inputs, outputs, masks, epoch, batch_idx, writer)
        # Calculate metrics
    return epoch_loss / len(loader)

def save_pretraining_mask_results(inputs, outputs, labels, epoch, batch_idx, writer=None):
    # 배치에서 첫 번째 이미지 사용
    input_image = inputs[0].cpu().detach()
    output_image = outputs[0].cpu().detach()
    label_image = labels[0].cpu().detach()
    target_size=(256, 256)

    label_image = label_image.float()

    if label_image.dim() == 2:
        label_image = label_image.unsqueeze(0)   # shape: [1, H, W]
    input_image = denormalize_rgb(input_image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    output_image    = denormalize_rgb(output_image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # label_image     = denormalize_rgb(label_image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    input_image = F.interpolate(input_image.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
    output_image = F.interpolate(output_image.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
    label_image = F.interpolate(label_image.unsqueeze(0).float(), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
    
    expanded_mask = label_image.repeat(3, 1, 1)
    output_image = (1 - expanded_mask) * input_image + expanded_mask * output_image
    # TensorBoard에 이미지 추가
    writer.add_image(f'Validation/Input_Epoch_{epoch}_Batch_{batch_idx}', input_image, epoch)
    writer.add_image(f'Validation/Predicted_Epoch_{epoch}_Batch_{batch_idx}', output_image, epoch)
    writer.add_image(f'Validation/GroundTruth_Epoch_{epoch}_Batch_{batch_idx}', label_image, epoch)

def save_pretraining_results(inputs, outputs, labels, epoch, batch_idx, writer=None):
    # 배치에서 첫 번째 이미지 사용
    input_image = inputs[0].cpu().detach()
    output_image = outputs[0].cpu().detach()
    label_image = labels[0].cpu().detach()
    target_size=(256, 256)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # 입력 이미지의 RGB 채널만 사용
    input_image_rgb = input_image[:3]  # [3, H, W]
    input_image_denorm  = denormalize_rgb(input_image_rgb, mean, std)
    input_image_denorm = F.interpolate(input_image_denorm.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
    # 시각화를 위해 [0, 1] 범위로 스케일링
    # input_image_rgb = (input_image_rgb - input_image_rgb.min()) / (input_image_rgb.max() - input_image_rgb.min())
    
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
    writer.add_image(f'Validation/Input_Epoch_{epoch}_Batch_{batch_idx}', input_image_denorm, epoch)
    writer.add_image(f'Validation/Predicted_Epoch_{epoch}_Batch_{batch_idx}', pred_mask, epoch)
    writer.add_image(f'Validation/GroundTruth_Epoch_{epoch}_Batch_{batch_idx}', true_mask, epoch)

def save_pretraining_denoise_result(inputs, outputs, labels, preds_label, epoch, batch_idx, writer = None):
    # 배치에서 첫 번째 이미지 사용
    input_image = inputs[0].cpu().detach()
    output_image = outputs[0].cpu().detach()
    label_image = labels[0].cpu().detach()
    preds_image = preds_label[0].cpu().detach()
    target_size=(256, 256)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    input_image_denorm  = denormalize_rgb(input_image, mean, std)
    denoised_image_denorm = denormalize_rgb(output_image, mean, std)
    label_image_denorm = denormalize_rgb(label_image, mean, std)
    preds_image_denorm = denormalize_rgb(preds_image, mean, std)
    # 텐서 형태 출력 (디버깅 용도)
    # print(f"Input Image Shape: {input_image_denorm.shape}")       # Expected: (3, H, W)
    # print(f"Denoised Image Shape: {denoised_image_denorm.shape}") # Expected: (3, H, W)
    # print(f"Label Image Shape: {label_image_denorm.shape}")       # Expected: (3, H, W)
    # print(f"Input Image Dtype: {input_image_denorm.dtype}")
    # print(f"Denoised Image Dtype: {denoised_image_denorm.dtype}")
    # print(f"Label Image Dtype: {label_image_denorm.dtype}")

    # 이미지 리사이즈 (CHW 형식이므로 배치 차원 추가 후 리사이즈 후 다시 제거)
    input_image_denorm = F.interpolate(input_image_denorm.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
    denoised_image_denorm = F.interpolate(denoised_image_denorm.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
    label_image_denorm = F.interpolate(label_image_denorm.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
    preds_image_denorm = F.interpolate(preds_image_denorm.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)

    writer.add_image(f'Validation/Input_Epoch_{epoch}_Batch_{batch_idx}', input_image_denorm, epoch, dataformats='CHW')
    writer.add_image(f'Validation/Predicted_Epoch_{epoch}_Batch_{batch_idx}', denoised_image_denorm, epoch, dataformats='CHW')
    writer.add_image(f'Validation/GroundTruth_Epoch_{epoch}_Batch_{batch_idx}', label_image_denorm, epoch, dataformats='CHW')
    writer.add_image(f'Validation/prednoise_Epoch_{epoch}_Batch_{batch_idx}', preds_image_denorm, epoch, dataformats='CHW')


def denormalize_rgb(tensor, mean, std):
    """
    정규화된 RGB 텐서를 역정규화하여 [0,1] 범위로 되돌립니다.

    Args:
        tensor (torch.Tensor): 정규화된 이미지 텐서 (C, H, W)
        mean (list or tuple): 각 채널의 평균값
        std (list or tuple): 각 채널의 표준편차

    Returns:
        torch.Tensor: 역정규화된 이미지 텐서 (C, H, W)
    """
    mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
    tensor_denorm = tensor * std + mean
    tensor_denorm = torch.clamp(tensor_denorm, 0.0, 1.0)
    return tensor_denorm


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
