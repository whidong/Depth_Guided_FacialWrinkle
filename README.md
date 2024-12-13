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
├── Masking_Face.ipynb                 # Face Masking
├── depth_map.ipynb                    # transform Depth map Gray scale
├── depth_masking.ipynb                # Masking Depth map
├── Merge_face_masking.ipynb           # Face masking RGB image
├── README.md 
```
