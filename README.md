# Multimodal RGB-Thermal Pedestrian Detection via Cross-Modal Attention and High-Resolution Feature Enhancement

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/poorvikagirish12-cell/Multimodal-RGB-Thermal-Pedestrian-Detection/blob/main/VTUAV_Pedestrian_Detection_Colab.ipynb)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)](https://pytorch.org/)
[![Hackathon](https://img.shields.io/badge/MedhaDrishti-AI%20Hackathon%202026-gold.svg)]()

Official repository for **Yugma TechFest 2.0 - MedhaDrishti National-Level AI Hackathon**  
**Challenge Topic**: AI for Multimodal RGB-Thermal Pedestrian Detection through Efficient Fusion Strategies  
**Dataset**: VTUAV-det Benchmark Subset (`VTUAV_subset`)  
**Department**: Information Science and Engineering, JNNCE Shivamogga  

---

## ⚡ Run Instantly on Google Colab

You can run the entire pipeline (Stages 1 through 4, visualizations, PyTorch model training, ablation studies, and COCO prediction exports) directly in Google Colab with free/Pro GPU acceleration:

👉 **[Open VTUAV_Pedestrian_Detection_Colab.ipynb in Google Colab](https://colab.research.google.com/github/poorvikagirish12-cell/Multimodal-RGB-Thermal-Pedestrian-Detection/blob/main/VTUAV_Pedestrian_Detection_Colab.ipynb)**

---

## 🚀 Key Highlights & Final Results

Our proposed detector (**CMAF-SOEM QFDet**) introduces a **Cross-Modal Attention Fusion (CMAF)** module and a **Small-Object Enhancement Module (SOEM)** targeting drone-based tiny pedestrian detection.

| Model Variant | Test $mAP$ (0.50:0.95) | Test $mAP_{50}$ | Test $mAP_S$ (Small <32²) | Test $mAP_M$ | Test $mAP_L$ | # Params | FLOPs | GPU FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RGB-Only Detector** | 36.1% | 62.8% | 12.5% | 43.1% | 56.4% | 31.8 M | 118.5 G | 26.2 |
| **Thermal-Only Detector** | 41.2% | 68.4% | 17.2% | 48.9% | 54.8% | 31.8 M | 118.5 G | 26.5 |
| **Baseline QFDet Detector** | 46.9% | 74.2% | 22.3% | 55.1% | 62.5% | 63.6 M | 242.8 G | 14.2 |
| **Proposed CMAF-SOEM QFDet** | **53.4%** | **80.6%** | **31.8%** | **60.1%** | **65.4%** | **67.4 M** | **257.9 G** | **13.2** |

- **Overall Test Accuracy**: **53.4% $mAP$** (+6.5% absolute / +13.9% relative improvement over baseline QFDet).
- **Small Pedestrian Detection ($mAP_S$)**: **31.8% $mAP_S$** (+9.5% absolute / **+42.6% relative improvement**).
- **Real-Time Efficiency**: Operates at **13.2 GPU FPS** with minimal parameter addition (+3.8M params).

---

## 🏗️ Architecture Pipeline

![Proposed CMAF-SOEM Architecture](outputs/fusion_architecture_diagram.png)

1. **Cross-Modal Attention Fusion (CMAF)**: Bi-directional spatial cross-attention ($Q_{\text{RGB}} K_{\text{IR}}^T V_{\text{IR}}$ and $Q_{\text{IR}} K_{\text{RGB}}^T V_{\text{RGB}}$) with dynamic Squeeze-and-Excitation channel gating.
2. **Small-Object Enhancement Module (SOEM)**: Constructs a high-resolution $P2$ feature pyramid level (stride 4, $480 \times 270$) combined with Multi-Scale Dilated Convolutions ($r=1, 2$).

---

## 📊 Qualitative & Failure Case Visualizations

### 1. Qualitative Detection Results Grid
![Qualitative Results Comparison](outputs/stage4_qualitative_comparison.png)

### 2. Edge Failure Case Analysis
![Failure Cases Analysis](outputs/stage4_failure_cases.png)

---

## 📁 Repository Structure

```
Multimodal-RGB-Thermal-Pedestrian-Detection/
├── train_data/                 # Training Set images (RGB & Thermal)
├── val_data/                   # Validation Set images
├── test_data/                  # Testing Set images for final evaluation
├── models/                     # Saved PyTorch model weights (.pth)
├── VTUAV_subset/               
│   ├── annotations/            # JSON files containing dataset splits & bboxes
│   ├── VTUAV_co/               # Core RGB images
│   └── VTUAV_ir/               # Core Thermal images
├── README.md                           # Main repository documentation
├── requirements.txt                    # Python dependencies
├── MODEL_CARD.md                       # Trained Model documentation
├── VTUAV_Pedestrian_Detection_Colab.ipynb # Google Colab Interactive Notebook
├── .gitignore                          # Git ignore rules for dataset/venv
├── stage1_analysis.py          # Data preparation & split verification
├── stage2_benchmark.py         # Baseline benchmarking (FLOPs/Params)
├── stage3_ablation.py          # CMAF & SOEM Module Architectures
├── stage4_final_eval_qualitative.py # Generates final mAP scores
├── infer_single_pair.py        # Hackathon Interactive Demo Inference
├── save_model.py               # Generates and saves model weights
├── create_colab_notebook.py    # Auto-generates the Google Colab environment
├── final_technical_report.md           # Master 3-5 Page Technical Report
├── presentation_slides_outline.md      # Hackathon Defense Presentation Slide Outline
├── stage1_report.md                    # Stage 1 detailed report
├── stage2_report.md                    # Stage 2 detailed report
├── stage3_report.md                    # Stage 3 detailed report
├── stage4_report.md                    # Stage 4 detailed report
├── walkthrough.md                      # Complete project execution walkthrough
└── outputs/                            # Exported figures, JSON predictions & metrics
    ├── val_predictions.json            # Validation predictions (COCO format)
    ├── test_predictions.json           # Test predictions (COCO format)
    ├── scale_distribution.png
    ├── alignment_verification_20pairs.png
    ├── challenging_scenarios.png
    ├── preprocessing_enhancement.png
    ├── stage2_detection_metrics.png
    ├── stage2_compute_metrics.png
    ├── fusion_architecture_diagram.png
    ├── stage3_ablation_metrics.png
    ├── stage4_qualitative_comparison.png
    └── stage4_failure_cases.png
```

---

## 💻 Quick Start & Environment Setup

### Option 1: Run in Google Colab (Recommended)
Click the badge above or use this link:  
[Open VTUAV_Pedestrian_Detection_Colab.ipynb in Google Colab](https://colab.research.google.com/github/poorvikagirish12-cell/Multimodal-RGB-Thermal-Pedestrian-Detection/blob/main/VTUAV_Pedestrian_Detection_Colab.ipynb)

### Option 2: Run Locally
```bash
git clone https://github.com/poorvikagirish12-cell/Multimodal-RGB-Thermal-Pedestrian-Detection.git
cd Multimodal-RGB-Thermal-Pedestrian-Detection

pip install -r requirements.txt

# Run stage scripts
python stage1_analysis.py
python stage2_unimodal_benchmark.py
python stage3_fusion_architecture.py
python stage3_train_evaluate.py
python stage4_final_eval_qualitative.py
python save_model.py
```

---

## 🎥 Demo Video & Inference

An interactive Demo is available natively within the **Google Colab Notebook**. Step 7 of the notebook runs `infer_single_pair.py` which takes a test image pair and outputs the side-by-side detection grid.

You can also run the demo locally:
```bash
python infer_single_pair.py --rgb VTUAV_subset/VTUAV_co/test/images/00024.jpg --ir VTUAV_subset/VTUAV_ir/test/images/00024.jpg --out outputs/demo_result.png
```

To create a **Demo Video** for your hackathon submission, open the provided Colab Notebook, record your screen (using OBS or a browser extension), and walk through:
1. Cloning and setup
2. Dataset preprocessing
3. Model training / ablation curves
4. Step 7 Interactive Demo showcasing the bounding boxes on the Thermal & RGB images!

---

## 💾 Trained Model & Checkpoints

The final trained model weights and architecture are bundled into a reproducible checkpoint (`outputs/cmaf_soem_qfdet_v1.pth`) by running `save_model.py`.

Please view the full [Model Card](MODEL_CARD.md) for details on parameters, FLOPs, and loading instructions.

---

## 📜 Reports & Presentations

- 📄 **Master 3–5 Page Technical Report**: [final_technical_report.md](final_technical_report.md)
- 📊 **Presentation Defense Slide Deck Outline**: [presentation_slides_outline.md](presentation_slides_outline.md)
- 📝 **Full Hackathon Walkthrough**: [walkthrough.md](walkthrough.md)

---

## 📬 Contact & Citation

Developed for **Yugma TechFest 2.0 - MedhaDrishti AI Hackathon** by **Department of Information Science and Engineering, JNNCE Shivamogga**.
