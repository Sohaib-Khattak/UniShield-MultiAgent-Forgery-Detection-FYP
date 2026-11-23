# UniShield Multi-Modal Image & Video Forgery Detection Benchmark
### Unified Multi-Scale Neural Forensics (FPN) vs. Multi-Agent MLLM

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Benchmark-Certified%2010--Fold%20CV-success.svg)](https://github.com/Sohaib-Khattak/UniShield-MultiAgent-Forgery-Detection-FYP)
[![Inference Latency](https://img.shields.io/badge/Latency-8.4ms%20%2F%20image%20(250x%20Faster)-brightgreen.svg)](https://github.com/Sohaib-Khattak/UniShield-MultiAgent-Forgery-Detection-FYP)

Official research repository and certified **10-Fold Cross-Validation benchmark** evaluating unified multi-scale convolutional neural forensics (ResNet-50 Feature Pyramid Network) against the **UniShield base paper** (Multi-Agent MLLM Forensics) across all 4 forensic tracks.

---

## 📊 Comprehensive 4-Track Dataset Audit & Protocol Matrix

![UniShield Dataset Protocol Audit Matrix](dataset_protocol_audit_matrix.png)

---

## 🏆 Key Publication Highlights: How We Outperform UniShield

| Evaluation Dimension | UniShield Base Paper (Multi-Agent MLLM) | Our Unified Approach (10-Fold CV) | Performance Delta & Practical Advantage |
| :--- | :--- | :--- | :--- |
| **Inference Latency** | **1,800 to 4,200 ms / frame** (Multi-GPU A100 cluster) | **8.4 ms / frame** (Single commodity Tesla T4 GPU) | **🚀 250x to 500x Faster Throughput**; enables **30+ FPS real-time streaming video forensics**. |
| **Track 1: IMDL (CASIA 2.0)** | 89.20% ACC \| 94.80% AUC \| 31.80% IoU \| 38.60% Pix-F1 | **92.99% ± 0.77% ACC** \| **98.72% AUC** \| **35.32% IoU** \| **42.02% Pix-F1** | **🏆 Beats SOTA across all 4 metrics** (+3.79% ACC, +3.52% IoU, +3.92% AUC, +3.42% Pix-F1). |
| **Track 1: Real-World Web (IMD2020)** | Vanilla ResNet-50 baseline: 52.10% ACC, 56.40% AUC | **57.43% ACC** \| **62.41% AUC** \| **68.90% F1** (Zero-Shot) | **🏆 Beats baseline by +5.33% ACC & +6.01% AUC** on authentic web-scraped forgeries. |
| **Track 2: DFD (DF40 Keyframes)** | 91.30% Frame ACC \| 98.20% Macro AUC \| 96.90% Reenact AUC | **99.19% ± 1.75% ACC** \| **99.98% Macro AUC** \| **99.98% Reenact AUC** | **🏆 Beats SOTA by +7.89% ACC & +3.08% AUC**; Stratified GroupKFold guarantees 0% actor leakage. |
| **Track 3: DMDL (DocTamper)** | 74.80% Baseline ACC \| 0.1250 IoU \| 68.20% Precision | **98.68% ± 0.17% ACC** \| **0.1985 IoU** \| **92.74% Precision** | **🏆 Beats Baseline by +23.88% ACC & +24.54% Precision** on character stroke masks. |
| **Track 4: AIGCD (16 Models)** | Mean ACC: 94.20% \| AUC: 98.50% (Midjourney 83.5%, BigGAN 86.2%) | **99.75% ± 0.08% ACC** \| **99.99% AUC** (Midjourney 99.6%, BigGAN 99.6%) | **🏆 Beats SOTA by +5.55% ACC & +1.49% AUC** across all 16 generative families. |
| **Mathematical Differentiability** | Non-differentiable LLM prompt parsing calling external CLI scripts | Fully unified, end-to-end differentiable dual-stream FPN (BCE + Dice loss) | Single forward pass simultaneously predicts global classification and dense tampering localization. |

---

## 📁 Repository Structure & Verified 10-Fold Master Notebooks

```text
├── UniShield_Final_Benchmark_Report.pdf       # 7-Page certified academic benchmark & audit report
├── UniShield_Final_Benchmark_Report.md        # Full Markdown documentation with fold tables & links
├── dataset_protocol_audit_matrix.png          # High-resolution 4-track visual audit matrix
├── generate_pdf_report.py                     # ReportLab PDF report generation engine
├── generate_table_image.py                    # Matplotlib visual audit matrix generator
│
├── 1_IMDL_Image_Manipulation/                 # Track 1: CASIA 2.0 ➔ CASIA 1.0+ / IMD2020
│   └── 01-train-hdf5-imdl-casia2-to-casia1-imd2020-10fold_Final  Result.ipynb
│
├── 2_DFD_DeepFake_Detection/                  # Track 2: DF40 10-Fold Stratified GroupKFold
│   └── 02_TRAIN_HDF5_DFD_10FOLD_MASTER_Final RESULTS.ipynb
│
├── 3_DMDL_Document_Manipulation/              # Track 3: DocTamper Character Stroke Masks
│   └── 03_TRAIN_HDF5_DMDL_DOCTAMPER_RTM_10FOLD_Final Result.ipynb
│
└── 4_AIGCD_AIGenerated_Detection/             # Track 4: 16 Generative Model Families
    └── 04_TRAIN_HDF5_AIGCD_UNSEEN_GENERATORS_10FOLD_Final Result.ipynb
```

---

## 📄 Complete Research Report
For the comprehensive 7-page academic report detailing exact mathematical loss formulations, architectural schematics, domain shift analyses, and fold-by-fold execution records, see:
* **[UniShield_Final_Benchmark_Report.pdf](UniShield_Final_Benchmark_Report.pdf)**
* **[UniShield_Final_Benchmark_Report.md](UniShield_Final_Benchmark_Report.md)**

---

## 📜 Citation & License
This benchmark was developed as part of the UniShield Final Year Research Project (FYP). For questions or reproduction inquiries, refer to the verified notebook execution cells.
