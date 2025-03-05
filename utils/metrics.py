import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import gc

def calculate_metrics_gpu(y_true, y_pred):
    """
    GPU를 사용하여 JSI, F1-Score, Accuracy, Precision, Recall 계산.
    """
    # Flatten the tensors
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Calculate TP, FP, FN, TN
    tp = ((y_pred == 1) & (y_true == 1)).sum().float()
    fp = ((y_pred == 1) & (y_true == 0)).sum().float()
    fn = ((y_pred == 0) & (y_true == 1)).sum().float()
    tn = ((y_pred == 0) & (y_true == 0)).sum().float()

    # JSI (IoU)
    jsi = tp / (tp + fp + fn + 1e-8)

    # Precision, Recall
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    # F1-Score
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Accuracy
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    del tp, fp, fn, tn
    gc.collect()

    return jsi.item(), f1.item(), acc.item(), precision.item(), recall.item()

# 전체 Depth 이미지에서 최소값과 최대값 계산
def calculate_depth_min_max(depth_paths):
    min_depth = float('inf')
    max_depth = float('-inf')

    for depth_path in depth_paths:
        depth_image = np.array(Image.open(depth_path).convert("L")).astype(np.float32)
        min_depth = min(min_depth, depth_image.min())
        max_depth = max(max_depth, depth_image.max())

    return min_depth, max_depth

def evaluate_model_gpu(loader, model, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)  # 다중 클래스 예측
                labels = labels.squeeze(1)  # [B, H, W] 형태로 변환

            # GPU에서 평가를 위해 리스트 대신 Tensor로 저장
            all_preds.append(preds)
            all_labels.append(labels)

    # Tensor로 결합
    all_preds = torch.cat(all_preds, dim=0).to(device)
    all_labels = torch.cat(all_labels, dim=0).to(device)

    # GPU에서 메트릭 계산
    jsi, f1, acc, precision, recall = calculate_metrics_gpu(all_labels, all_preds)
    return jsi, f1, acc, precision, recall

def add_noise(x, noise_type = 'simple', noise_std = 0.22):

    noise = torch.randn_like(x)

    if noise_type == "simple":
        x_noise = x + noise * noise_std
    elif noise_type == "scaled":
        x_noise = ((1 + noise_std**2) ** -0.5) * (x + noise * noise_std)
    return x_noise, noise

