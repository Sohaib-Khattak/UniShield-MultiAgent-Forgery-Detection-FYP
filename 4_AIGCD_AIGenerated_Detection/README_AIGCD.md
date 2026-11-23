# UniShield Domain 4: AI-Generated Content Detection (AIGCD)

This module implements the **AIGCD** track from the CVPR Findings 2026 paper:
> **"UniShield: An Adaptive Multi-Agent Framework for Unified Forgery Image Detection and Localization"**

---

## 1. Datasets Used in Paper
- **AIGCDetectBenchmark (Zhong et al. 2023)**: Evaluated across **16 leading generative models**:
  1. ProGAN
  2. StyleGAN
  3. BigGAN
  4. CycleGAN
  5. StarGAN
  6. GauGAN
  7. StyleGAN2
  8. WFTIR
  9. ADM
  10. Glide
  11. Midjourney
  12. Stable Diffusion v1.4 (SD v1.4)
  13. Stable Diffusion v1.5 (SD v1.5)
  14. VQDM
  15. Wukong
  16. DALL-E 2 (DALLE2)

### Kaggle Dataset Links:
- **AI vs Human Generated Images (Configured)**: [https://www.kaggle.com/datasets/sohaibkhatak/ai-vs-human-generated-images](https://www.kaggle.com/datasets/sohaibkhatak/ai-vs-human-generated-images)
  *(Contains both `train` and `test` data with `AI` and `Human` folders)*
- [CIFAKE: Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- [AI-Generated Images Dataset on Kaggle](https://www.kaggle.com/datasets/swetha280/ai-generated-images-dataset)
- [AI vs Human Generated Dataset](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset)

---

## 2. Models Used in Paper (Table 1 & Table 6)
- **Non-LLM Expert**: `AIDE` [53] (Combines patch-level spatial features and 2D Fourier spectrum frequency artifacts).
- **LLM-Based Expert**: `FakeVLM` [47] (Spot the Fake multimodal artifact explanation agent).
- **Perception Agent**:
  - Task Router routes whole-image generation to `AIGCD`.
  - Tool Scheduler adaptively selects between `AIDE` (high-frequency grid/checkerboard upsampling traces) and `FakeVLM` (semantic over-smoothing and perspective violations).

---

## 3. Paper Reported Performance on AIGCDetectBenchmark (Table 6)
| Generative Model | CNNSpot | FreDect | UnivFD | DIRE | PatchCraft | AIDE | FakeVLM | **UniShield (Ours)** |
|---|---|---|---|---|---|---|---|---|
| ProGAN | 1.000 | 0.994 | 0.998 | 0.528 | 1.000 | 1.000 | 0.997 | **0.999** |
| StyleGAN | 0.902 | 0.780 | 0.849 | 0.513 | 0.928 | 0.996 | 0.917 | **0.998** |
| BigGAN | 0.712 | 0.820 | 0.951 | 0.497 | 0.958 | 0.840 | 0.782 | **0.862** |
| CycleGAN | 0.876 | 0.788 | 0.983 | 0.496 | 0.702 | 0.985 | 0.855 | **0.991** |
| StarGAN | 0.946 | 0.946 | 0.958 | 0.467 | 1.000 | 0.999 | 0.991 | **1.000** |
| GauGAN | 0.814 | 0.806 | 0.995 | 0.512 | 0.716 | 0.733 | 0.749 | **0.783** |
| StyleGAN2 | 0.869 | 0.662 | 0.750 | 0.517 | 0.850 | 0.980 | 0.958 | **0.973** |
| WFTIR | 0.917 | 0.508 | 0.869 | 0.533 | 0.822 | 0.942 | 0.713 | **0.958** |
| ADM | 0.604 | 0.634 | 0.669 | 0.983 | 0.838 | 0.934 | 0.840 | **0.948** |
| Glide | 0.581 | 0.541 | 0.625 | 0.924 | 0.901 | 0.951 | 0.649 | **0.956** |
| Midjourney | 0.514 | 0.459 | 0.561 | 0.895 | 0.954 | 0.772 | 0.660 | **0.835** |
| SD v1.4 | 0.506 | 0.388 | 0.637 | 0.912 | 0.953 | 0.930 | 0.653 | **0.959** |
| SD v1.5 | 0.505 | 0.392 | 0.635 | 0.916 | 0.889 | 0.929 | 0.850 | **0.939** |
| VQDM | 0.565 | 0.778 | 0.853 | 0.919 | 0.911 | 0.952 | 0.673 | **0.961** |
| Wukong | 0.510 | 0.403 | 0.709 | 0.909 | 0.966 | 0.936 | 0.810 | **0.946** |
| DALLE2 | 0.505 | 0.347 | 0.508 | 0.924 | 0.893 | 0.966 | 0.864 | **0.970** |
| **Mean Across 16** | 0.708 | 0.640 | 0.784 | 0.715 | 0.893 | 0.928 | 0.810 | **0.942** |

---

## 4. How to Run
### On Kaggle (Recommended 5-Fold Protocol):
1. Open and upload [UniShield_AIGCD_Kaggle_5Fold.ipynb](file:///c:/Users/ADNANLAPTOP/Desktop/Code%20for%20dataset/4_AIGCD_AIGenerated_Detection/UniShield_AIGCD_Kaggle_5Fold.ipynb).
2. Attach the dataset **`AI vs. Human-Generated Images`** (`alessandrasala79/ai-vs-human-generated-dataset`) via **'+ Add Input'**.
3. Select Accelerator: **GPU T4 x2** (Settings $\rightarrow$ Accelerator $\rightarrow$ **GPU T4 x2**).
4. In Cell 2:
   - `MAX_SAMPLES = None` trains on all **~80,000 images** (~60 min on dual T4).
   - Set `MAX_SAMPLES = 10000` for a fast initial verification run.
5. Click **Run All**. The notebook will automatically crawl `/kaggle/input`, discover `train.csv`, index the real and AI images, and run 5-Fold Cross-Validation.

### In Terminal / Command Line:
```bash
# 1. Run 5-Fold CV with attached Kaggle dataset:
python unishield_aigcd.py --data_dir /kaggle/input/ai-vs-human-generated-dataset --split all --epochs 5 --folds 5

# 2. Or run on local data directory:
python unishield_aigcd.py --data_dir ./data/AIGCD --split all --epochs 5 --folds 5
```
Outputs are written to `/kaggle/working/` (or `./outputs/AIGCD/`).
