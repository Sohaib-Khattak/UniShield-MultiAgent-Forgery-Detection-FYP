# UniShield Domain 2: DeepFake Face Forgery Detection (DFD)

This module implements the **DFD** track from the CVPR Findings 2026 paper:
> **"UniShield: An Adaptive Multi-Agent Framework for Unified Forgery Image Detection and Localization"**

---

## 1. Datasets Used in Paper
- **DF40 Benchmark**: A comprehensive dataset covering 40 face forgery techniques, divided into:
  - **FS (Face Swap)**: E.g., FaceShifter, SimSwap, DeepFaceLab.
  - **FR (Face Reenactment)**: E.g., FOMM, Wav2Lip, LivePortrait.
- **FaceForensics++ (FF++)** & Celeb-DF.

### Kaggle Dataset Links:
- [Deepfake and Real Images on Kaggle (manjilkarki/deepfake-and-real-images)](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)
- [FaceForensics++ on Kaggle (greatgamedota/faceforensics)](https://www.kaggle.com/datasets/greatgamedota/faceforensics)
- [Deepfake Detection Challenge Dataset](https://www.kaggle.com/competitions/deepfake-detection-challenge/data)

---

## 2. Model Backbones
- **ResNet-50 (Default / Supervisor Choice)**:
  - `torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)` (ImageNet-1K V2, 80.86% top-1 accuracy).
  - Fine-tuned with custom binary classification head: `Dropout(0.3) -> Linear(2048, 256) -> ReLU -> Dropout(0.2) -> Linear(256, 1)`.
  - Directly matches the baseline evaluated in Table 4 of the CVPR 2026 paper.
- **DFD Expert**: Dual-stream Spatial RGB + High-Frequency DCT/Laplacian branch with Cross-Modal Attention Fusion.
- **DFD-R1**: Fine-tuned MLLM with GRPO outcome-based reasoning.

---

## 3. Paper Reported Performance (Table 4)
| Method | AUC (FS Subset) | AUC (FR Subset) |
|---|---|---|
| Xception | 0.991 | 0.892 |
| CLIP | 0.996 | 0.908 |
| SRM | 0.988 | 0.867 |
| SPSL | 0.987 | 0.849 |
| RECCE | 0.991 | 0.855 |
| RFM | 0.992 | 0.884 |
| DFD-R1 | 0.923 | 0.897 |
| **UniShield (Ours)** | **0.997** | **0.913** |

---

## 4. How to Run

### On Kaggle:
1. Upload and open [UniShield_DFD_Kaggle_GPU_5Fold.ipynb](file:///c:/Users/ADNANLAPTOP/Desktop/Code%20for%20dataset/2_DFD_DeepFake_Detection/UniShield_DFD_Kaggle_GPU_5Fold.ipynb).
2. Attach your dataset via Kaggle's **'+ Add Data'**:
   - If using your uploaded **UniShield DF40** dataset (`sohaibkhatak/unisheild-df40`), Kaggle mounts it under `/kaggle/input/unisheild-df40` or `/kaggle/input/unishield-df40`.
   - The notebook automatically auto-detects `UniShield DF40` from `/kaggle/input` dynamically!
   - You can also manually set `MANUAL_DATA_DIR = "/kaggle/input/unisheild-df40"` in Section 1.
3. Select Accelerator: **GPU T4 x2** or **GPU P100** (under Settings ⋮ in the top right panel).
4. Click **Run All**.
   - The notebook discovers Real faces from `CelebA-HQ-img` (30,000 images) and `FaceForensics++` (31,949+ frames).
   - It automatically extracts representative face frames from `Celeb-DF` FaceSwap fake videos into `/kaggle/working/dfd_extracted_fake_frames/`.
   - It performs 5-Fold Cross-Validation (80/10/10 Train/Val/Test in each fold, 5 epochs per fold) with ImageNet-1K V2 ResNet-50.

### In Terminal:
```bash
# Run ResNet-50 model with 10-fold cross validation (5 epochs per fold):
python 2_DFD_DeepFake_Detection/unishield_dfd.py --model resnet50 --epochs 5 --folds 10

# Or run with Frequency/CLIP expert:
python 2_DFD_DeepFake_Detection/unishield_dfd.py --model clip --epochs 5 --folds 10
```
Outputs are written to `./outputs/DFD/`.

