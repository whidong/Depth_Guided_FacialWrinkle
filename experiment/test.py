import sys
import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.append("../")  # 프로젝트 루트 추가
from model import create_model
from datasets.dataset import WrinkleDataset, WrappedDataset
from utils.metrics import evaluate_model_gpu, calculate_depth_min_max
from utils.train_utils import load_test
import albumentations as A
from albumentations.pytorch import ToTensorV2

def test_pretrained_model(run_id, seed, mode, model_type, batch_size=10, pretrain_path=None):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(42)

    print(f"Testing: run_id={run_id}, seed={seed}, mode={mode}, model={model_type}")

    if model_type == "custom_unet":
        in_ch = {"RGB": 3, "RGBT": 4,"RGB_check": 3, "RGBT_check": 4, "RGB_origin":3, "RGBT_origin":4,"RGBDT_origin": 5,"RGBDT_full" : 5, "RGBDT": 5, "RGBDT_test": 5, "denoise": 3}.get(mode, None)
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
            "RGB_no": (3, 2, 48, (2, 2, 2, 2), False),
            "RGBT": (4, 2, 48, (2, 2, 2, 2), False),
            "RGBT_origin": (4, 2, 48, (2, 2, 2, 2), False),
            "RGBDT": (5, 2, 48, (2, 2, 2, 2), False),
            "RGBDT_origin": (5, 2, 48, (2, 2, 2, 2), False),
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

    elif model_type == "strip_net":
        in_ch = {"RGB" : 3}.get(mode, None)
        out_ch = 2
        if in_ch is None:
            raise ValueError(f"Unknown mode: {mode}")

        model = create_model(
            model_type = model_type,
            in_channels = in_ch,
            out_channels = out_ch
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    model = nn.DataParallel(model).cuda()

    if pretrain_path:
        model = load_test(model, pretrain_path)

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

    data_list = list(zip(rgb_paths, depth_paths, weak_texture_paths, label_paths))
    random.seed(42)
    random.shuffle(data_list)
    rgb_paths, depth_paths, weak_texture_paths, label_paths = zip(*data_list)

    min_depth, max_depth = calculate_depth_min_max(depth_paths)
    dataset = WrinkleDataset(rgb_paths, depth_paths, weak_texture_paths, label_paths,
                             transform=None, min_depth=min_depth, max_depth=max_depth, mode=mode, task='finetune')
    # 데이터 분할
    total_len = len(dataset)
    train_size = int(0.8 * total_len)
    val_size   = int(0.1 * total_len)
    test_size = total_len - train_size - val_size
    train_subset, val_subset, test_subset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406, 0, 0), std=(0.229, 0.224, 0.225, 1, 1)),
        ToTensorV2(transpose_mask=True)
    ])

    test_dataset = WrappedDataset(test_subset, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # evaluate
    jsi, f1, acc, prec, rec = evaluate_model_gpu(test_loader, model)
    print(f"[Mode={mode}, Model={model_type}, Seed={seed}] Pretrain={pretrain_path}")
    print(f"  => JSI={jsi:.5f}, F1={f1:.5f}, Acc={acc:.5f}, Prec={prec:.5f}, Rec={rec:.5f}")
    return jsi, f1, acc, prec, rec




def main():
    run_seeds = [42, 2025, 2024]
    experiments = {
        "custom_unet": {"modes": {"RGBDT_origin": "RGBDT_origin","RGBT_origin": "RGBT_origin","RGB_origin": "RGB_origin","RGBT_check": "RGBT_check","RGB_check": "RGB_check"}},

        "custom_unetr": {"modes": {"RGBT_origin" : "RGBT_origin" }},

    }
    batch_size = 14
    for model_type, settings in experiments.items():
        print(f"\n=== Evaluating pretrained models: {model_type} ===")
        for mode in settings["modes"]:
            results = []
            for run_id, seed in enumerate(run_seeds):
                pretrain_path = f"./checkpoint/fine_tuning/{model_type}_{mode}/best_{model_type}_{mode}_{run_id}_seed{seed}.pth"
                jsi, f1, acc, prec, rec = test_pretrained_model(run_id, seed, mode, model_type, batch_size, pretrain_path)
                results.append((jsi, f1, acc, prec, rec))
            results_array = np.array(results)
            means = results_array.mean(axis=0)
            stds = results_array.std(axis=0)
            metric_names = ["JSI", "F1", "Acc", "Prec", "Recall"]
            print(f"\n==== Mode={mode} Final Results across seeds={run_seeds} ====")
            for i, mname in enumerate(metric_names):
                print(f"{mname}: {means[i]:.5f} ± {stds[i]:.5f}")

if __name__ == "__main__":
    main()
