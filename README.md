## Facial Wrinkle Segmentation with Combined Depth-Guided Feature Sampling and Texture Map weak Supervision

Masking Depth map Dataset : https://drive.google.com/drive/u/1/folders/1hHxali_6WGRa65ttyOdcFnq0Ad1f9hu0

Depth Estimation Model(PatchRefiner) : https://github.com/zhyever/PatchRefiner

FFHQ face wrinkle Dataset : https://github.com/labhai/ffhq-wrinkle-dataset


File Structure
```
├── model  
  ├── __init__.py
  ├── segmentation_models.py
  ├── U_net.py
  ├── Custom_UNet.py
  ├── swin_transformer.py
  ├── swin_transformer_v2.py
  ├── Custom_swin_unetr.py
  ├── swin_unetr.py
├── experiments
  ├── train.py                     
  ├── pretrain.py      
  ├── test.py
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
