# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

import math
import random
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import ImageFolder
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask


class SimMIMTransform:
    def __init__(self, input_size=1024, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6, val=False):
        if val:
            self.transform_img = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size)),
                T.ToTensor(),
                T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD)),
            ])
        else:
            self.transform_img = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(input_size, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD)),
            ])
        
        self.mask_generator = MaskGenerator(
            input_size=input_size,
            mask_patch_size= mask_patch_size,
            model_patch_size= model_patch_size,
            mask_ratio=mask_ratio,
        )
    
    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()
        
        return img, mask

class MySimMIMDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img_pil = Image.open(path).convert("RGB")
        if self.transform:
            img_tensor, mask_array = self.transform(img_pil)
            return img_tensor, mask_array
        else:
            return img_pil, None

def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret

def build_loader_simmim(
    data_path, 
    batch_size=8, 
    num_workers=4, 
    input_size=1024,
    mask_patch_size=32,
    model_patch_size=4,
    mask_ratio=0.6,
    shuffle=True,
    drop_last=True,
    validation=False,
):
    """
    Args:
        data_path (str): ImageFolder 형식의 이미지가 저장된 상위 디렉토리 경로.
        batch_size (int): 배치 크기
        num_workers (int): DataLoader num_workers
        input_size (int): RandomResizedCrop 및 마스크 생성 시 사용할 이미지 크기
        mask_patch_size (int)
        model_patch_size (int)
        mask_ratio (float)
        distributed (bool): DDP(분산학습) 사용 여부
        shuffle (bool): 샘플 순서 섞기
        drop_last (bool): 마지막 배치 버릴지 여부

    Returns:
        DataLoader: (img, mask) 형태를 출력하는 DataLoader
    """

    validation = validation
    if validation:
        transform = SimMIMTransform(
        input_size=input_size,
        mask_patch_size=mask_patch_size,
        model_patch_size=model_patch_size,
        mask_ratio=mask_ratio,
        val = True
    )
    else:
        transform = SimMIMTransform(
        input_size=input_size,
        mask_patch_size=mask_patch_size,
        model_patch_size=model_patch_size,
        mask_ratio=mask_ratio,
        val = False
        )

    # 2) Dataset (ImageFolder)
    dataset = MySimMIMDataset(data_path, transform=transform)
    print(f"[build_loader_simmim] # of images = {len(dataset)}")

    # 3) DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle= shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate_fn
    )
    
    return dataloader