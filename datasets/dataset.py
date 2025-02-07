import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class WrinkleDataset(Dataset):
    def __init__(self, rgb_paths, depth_paths=None, weak_texture_paths=None, label_paths=None, transform=None, 
                 min_depth=55.0, max_depth=247.0, mode="RGBDT", task = "pretrain"):
        """
        Args:
            rgb_paths (str): RGB 이미지 경로.
            depth_paths (str): Depth 맵 경로.
            weak_texture_paths (str): Weak Texture Map 경로.
            label_paths (str): Masked Texture Map 경로.
            transform (callable, optional): 데이터 변환 함수.
            mode (str): 입력 모드 선택 ("RGB", "RGBD", "RGBT", "RGBDT").
        """
        self.rgb_paths = rgb_paths
        self.depth_paths = depth_paths if depth_paths is not None else []
        self.weak_texture_paths = weak_texture_paths  if weak_texture_paths is not None else []
        self.label_paths = label_paths if label_paths is not None else []
        self.transform = transform

        # RGB 이미지 정규화를 위한 평균과 표준편차
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        
        # Depth 이미지 정규화를 위한 최소값과 최대값
        self.min_depth = min_depth
        self.max_depth = max_depth

        # 입력 모드 설정
        self.mode = mode
        self.task = task

    def __len__(self):
        return len(self.rgb_paths)

    def set_transform(self, transform):
        """
        데이터셋의 변환(transform)을 동적으로 변경하는 메서드.
        """
        self.transform = transform

    def __getitem__(self, idx):
        rgb_path = self.rgb_paths[idx]
        weak_texture_path = self.weak_texture_paths[idx] if self.weak_texture_paths else None
        depth_path = self.depth_paths[idx] if self.depth_paths else None
        label_path = self.label_paths[idx]

        rgb_image = np.array(Image.open(rgb_path).convert("RGB")).astype(np.float32)

        if self.task == "pretrain":
            #pretrain label is weak texture map
            if self.mode == "denoise":
                label = rgb_image.copy()
            else:
                label = np.array(Image.open(label_path).convert("L")).astype(np.float32)
                label = np.expand_dims(label, axis = -1)

        elif self.task == "finetune":
            label = np.array(Image.open(label_path).convert("L")).astype(np.int64)  # 정수형으로 변환
            label = (label > 0.5).astype(np.int64) 
            
        # Depth 로드 및 정규화
        if "D" in self.mode:
            depth_image = np.array(Image.open(depth_path).convert("L")).astype(np.float32)
            depth_image = (depth_image - self.min_depth) / (self.max_depth - self.min_depth)
            depth_image = np.clip(depth_image, 0, 1)
            depth_image = np.expand_dims(depth_image, axis=-1)  # (H, W, 1)
        
        # Weak Texture 로드 및 정규화
        if "T" in self.mode:
            weak_texture_image = np.array(Image.open(weak_texture_path).convert("L")).astype(np.float32)
            weak_texture_image = np.expand_dims(weak_texture_image, axis=-1)  # (H, W, 1)

        # 입력 데이터 생성
        input_image = rgb_image
        if "D" in self.mode:
            input_image = np.concatenate((input_image, depth_image), axis=-1)
        if "T" in self.mode:
            input_image = np.concatenate((input_image, weak_texture_image), axis=-1)

        #print(f"[WrinkleDataset] idx={idx}, AFTER LOAD: input_image={input_image.shape} (type={input_image.dtype}), label={label.shape} (type={label.dtype})")

        # Transform 적용
        if self.transform:
            augmented = self.transform(image=input_image, mask=label)
            input_image = augmented['image']
            label = augmented['mask']
            #print(f"[WrinkleDataset] idx={idx}, AFTER Albumentations: input={input_image.shape} type={input_image.dtype}, label={label.shape} type={label.dtype}")
        else:
            # ToTensor 변환
            input_image = ToTensorV2()(image=input_image)['image']
            label = ToTensorV2()(image=label)['image']
            #print(f"[WrinkleDataset] idx={idx}, AFTER ToTensorV2: input={input_image.shape} type={input_image.dtype}, label={label.shape} type={label.dtype}")
        # print(f"Before Normalize: min={rgb_image.min()}, max={rgb_image.max()}")
        return input_image, label

def get_train_augmentations():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
        A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.2),
        A.Affine(scale=(0.8, 1.2), translate_percent=(0.1, 0.1), rotate=(-15, 15), shear=(-10, 10), p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # RGB + Depth + Weak Texture
        ToTensorV2(transpose_mask=True),
    ])

def get_pre_augmentations():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),  # 'affine' 인자 제거
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
        A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.2),
        A.Affine(scale=(0.8, 1.2), translate_percent=(0.1, 0.1), rotate=(-15, 15), shear=(-10, 10), p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Depth 채널 포함
        ToTensorV2(transpose_mask=True),
    ])

def get_texture_augmentations():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),  # 'affine' 인자 제거
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
        A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.2),
        A.Affine(scale=(0.8, 1.2), translate_percent=(0.1, 0.1), rotate=(-15, 15), shear=(-10, 10), p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Depth 채널 포함
        ToTensorV2(transpose_mask=True),
    ])


class WrappedDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        try:
            # 데이터셋에서 이미지와 레이블을 로드합니다.
            
            image, label = self.subset[idx]
            #print(f"[WrappedDataset] idx={idx}, BEFORE ANY CHANGES: image={type(image)}, label={type(label)}")
            # print(f"Original label shape: {label.shape}")  # 디버깅 로그
            #if isinstance(image, torch.Tensor):
            #    print(f"   image.shape={image.shape}, dtype={image.dtype}")
            #if isinstance(label, torch.Tensor):
            #    print(f"   label.shape={label.shape}, dtype={label.dtype}")
            # 이미지가 torch.Tensor인 경우, numpy 배열로 변환
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
                # print(f"[WrappedDataset] idx={idx}, after to numpy: shape={image.shape}, dtype={image.dtype}")
                # 채널 순서를 (C, H, W)에서 (H, W, C)로 변경
                image = np.transpose(image, (1, 2, 0))
                # print(f"[WrappedDataset] idx={idx}, after transpose: shape={image.shape}, dtype={image.dtype}")

            # 레이블이 torch.Tensor인 경우, numpy 배열로 변환
            if isinstance(label, torch.Tensor):
                label = label.cpu().numpy()
                # print(f"[WrappedDataset] idx={idx}, label to numpy: shape={label.shape}, dtype={label.dtype}")
                # 채널 순서를 (C, H, W)에서 (H, W, C)로 변경
                label = np.transpose(label, (1, 2, 0))
                # print(f"[WrappedDataset] idx={idx}, label after transpose: shape={label.shape}, dtype={label.dtype}")

            
            # 이미지와 레이블이 PIL Image인 경우, numpy 배열로 변환
            if isinstance(image, Image.Image):
                image = np.array(image)
                
            if isinstance(label, Image.Image):
                label = np.array(label)

            # print(f"Label shape before transform: {label.shape}")  # 디버깅 로그
            
            # albumentations 변환 적용
            if self.transform:
                augmented = self.transform(image=image, mask=label)
                image = augmented['image']
                label = augmented['mask']
                # print(f"[WrappedDataset] idx={idx}, AFTER Albumentations: image.shape={image.shape}, label.shape={label.shape}")
            # print(f"Label shape after transform: {label.shape}")  # 디버깅 로그
            
            # 레이블이 다중 채널인 경우, 단일 채널로 변환
            if isinstance(label, torch.Tensor):
                if label.dim() == 3:
                    if label.shape[0] > 1:
                        # (C,H,W), C>1 → argmax
                        label = torch.argmax(label, dim=0)  # shape=(H,W)
                    else:
                        # (1,H,W) → squeeze -> (H,W)
                        label = label.squeeze(0)
            # else if label.dim()==2, already (H,W)
    
            if isinstance(label, torch.Tensor):
                # 1) 라벨이 텐서인 경우
                # dtype 보정
                if label.dtype != torch.int64:
                    label = label.long()
                    # shape 조정 (dim, squeeze/argmax 등)

            elif isinstance(label, np.ndarray):
                # 2) 라벨이 np.ndarray인 경우
                if label.ndim == 3:
                    if label.shape[2] > 1:
                        label = np.argmax(label, axis=2)
                    else:
                        label = label.squeeze(2)
                label = torch.from_numpy(label).long()

            else:
                # 3) 둘 다 아니면 에러
                raise ValueError(f"Unexpected label type: {type(label)}")

            
            # print(f"[WrappedDataset] idx={idx}, FINAL label: shape={label.shape}, dtype={label.dtype}")
            # print(f"Final label shape: {label.shape}")  # 디버깅 로그
            
            return image, label
        
        except Exception as e:
            print(f"Error in __getitem__ for index {idx}: {e}")
            raise e