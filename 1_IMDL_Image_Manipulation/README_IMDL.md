# UniShield Domain 1: Image Manipulation Detection and Localization (IMDL)

This module implements the **IMDL** track from the CVPR Findings 2026 paper:
> **"UniShield: An Adaptive Multi-Agent Framework for Unified Forgery Image Detection and Localization"**  
> *Qing Huang, Zhipei Xu, Xuanyu Zhang, Xiangyu Yu, Jian Zhang (Peking University, SCUT)*

---

## 1. Datasets Used in Paper
1. **CASIA1+ (CASIA 1.0 & CASIA 2.0)**: Benchmark natural image tampering dataset containing copy-move, splicing, and removal manipulations with corresponding ground-truth binary masks.
2. **IMD2020**: Real-world manipulated image benchmark with fine-grained pixel-level annotations.

### Kaggle Dataset Links & Loading Area:
- **CASIA 2.0 on Kaggle (sophatvathana/casia-dataset)**: [https://www.kaggle.com/datasets/sophatvathana/casia-dataset](https://www.kaggle.com/datasets/sophatvathana/casia-dataset)
- **IMD2020 on Kaggle (defileroff/imd2020-image-manipulation-dataset)**: [https://www.kaggle.com/datasets/defileroff/imd2020-image-manipulation-dataset](https://www.kaggle.com/datasets/defileroff/imd2020-image-manipulation-dataset)
- **Kaggle Input Directory**: `/kaggle/input/casia-dataset/CASIA2` or `/kaggle/input/imd2020-image-manipulation-dataset`

> *Note: If run without an attached dataset, the code automatically activates a synthetic benchmark generator with realistic spliced patches and masks to ensure out-of-the-box execution.*

---

## 2. Model Backbones
- **ResNet-50 (Default / Supervisor Choice)**:
  - `torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)` (ImageNet-1K V2, 80.86% top-1 accuracy).
  - **Dual-Head Architecture**:
    1. Global Average Pooling + Classification MLP -> Tampering probability (`cls_logit`).
    2. Multi-Scale Feature Pyramid Network (FPN) Decoder (`layer4` down through `layer1`) -> Pixel tampering mask (`mask_logit` $B \times 1 \times H \times W$).
  - Directly matches the baseline evaluated in Table 3 of the CVPR 2026 paper.
- **IML-ViT**: Vision Transformer-inspired multi-scale patch encoder with boundary fusion.
- **FakeShield**: Multi-modal cross-modal grounding for semantic inconsistency.

---

## 3. Paper Reported Performance (Tables 2 & 3)
| Benchmark | Method | ACC | Img-F1 | IoU | Pix-F1 |
|---|---|---|---|---|---|
| **CASIA1+** | FakeShield | 0.95 | 0.95 | 0.54 | 0.60 |
| **CASIA1+** | IML-ViT | 0.92 | 0.93 | 0.69 | 0.76 |
| **CASIA1+** | **UniShield (Ours)** | **0.97** | **0.96** | **0.70** | **0.77** |
| **IMD2020** | FakeShield | 0.83 | 0.90 | 0.50 | 0.57 |
| **IMD2020** | IML-ViT | 0.83 | 0.90 | 0.30 | 0.39 |
| **IMD2020** | **UniShield (Ours)** | **0.84** | **0.91** | **0.51** | **0.60** |

---

## 4. How to Run
### Option A: Interactive Kaggle Notebook
1. Upload [UniShield_IMDL_Kaggle.ipynb](file:///c:/Users/ADNANLAPTOP/Desktop/Code%20for%20dataset/1_IMDL_Image_Manipulation/UniShield_IMDL_Kaggle.ipynb) to your Kaggle account.
2. In the right panel, attach the CASIA or IMD2020 dataset (via "+ Add Data").
3. Select Accelerator: **GPU P100** or **GPU T4 x2**.
4. Click **Run All**.

### Option B: Standalone Python Script
```bash
# Run ResNet-50 model with 10-fold cross validation (5 epochs per fold):
python 1_IMDL_Image_Manipulation/unishield_imdl.py --model resnet50 --epochs 5 --folds 10

# Or run with IML-ViT:
python 1_IMDL_Image_Manipulation/unishield_imdl.py --model iml_vit --epochs 5 --folds 10
```
Outputs (checkpoints, ROC curve plots, 10-fold summary table, and sample forensic reports) will be saved in `./outputs/IMDL/`.
