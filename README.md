## Facial Wrinkle Segmentation with Combined Depth-Guided Feature Sampling and Texture Map weak Supervision

Masking Depth map Dataset : https://drive.google.com/drive/u/1/folders/1hHxali_6WGRa65ttyOdcFnq0Ad1f9hu0

Depth Estimation Model(PatchRefiner) : https://github.com/zhyever/PatchRefiner


File Structure
```
├── model  
  ├── __init__.py
  ├── U_net.py
  ├── swin_unetr.py
├── experiments
  ├── U_net.ipynb                      # Pretraining
  ├── Unet_fine_final.ipynb            # Finetuning
  ├── trained_unet_full_checkpoint.pth # Pretrain checkpoint
├── data
  ├── depth_masking
  ├── finetuning
    ├── depth_masking
    ├── manual_wrinkle_masks
    ├── masked_face_images
    ├── weak_wrinkle_mask
  ├── images 1024x1024
  ├── manual_wrinkle_masks
  ├── weak_wrinkle_image
  ├── weak_wrinkle_masks
├── datasets
  ├── __init__.py
  ├── dataset.py
├── utils
  ├── __init__.py
  ├── custom_scheduler.py
  ├── metrics.py
  ├── train_utils.py
├── loss
  ├── dice_loss
├── README.md 
```
experiment - ing ~
