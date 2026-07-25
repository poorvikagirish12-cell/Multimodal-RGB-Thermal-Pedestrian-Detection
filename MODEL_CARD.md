# CMAF-SOEM QFDet — Model Card

## Overview
**Model Name**: CMAF-SOEM QFDet v1.0.0  
**Hackathon**: Yugma TechFest 2.0 - MedhaDrishti AI Hackathon  
**Dataset**: VTUAV-det Subset (1700 RGB-Thermal pairs, 12543 pedestrian annotations)  
**Task**: Pedestrian Detection (class: `person`) — RGB-Thermal Multimodal  

---

## Architecture

| Component | Details |
|:---|:---|
| **Backbone** | Dual ResNet-50 (RGB + Thermal streams) |
| **Fusion Module** | Cross-Modal Attention Fusion (CMAF) — bi-directional spatial attention with SE gating |
| **Neck** | Small-Object Enhancement Module (SOEM) — P2 FPN level with multi-scale dilated convolutions |
| **Detection Head** | Quality-aware focal detection head |

---

## Performance (COCO Metrics on Test Set)

| Metric | Baseline QFDet | **Proposed CMAF-SOEM** | Improvement |
|:---|:---:|:---:|:---:|
| mAP (0.50:0.95) | 46.9% | **53.4%** | +6.5% abs / +13.9% rel |
| mAP50 | 74.2% | **80.6%** | +6.4% abs |
| **mAPS (Small <32²)** | 22.3% | **31.8%** | **+9.5% abs / +42.6% rel** |
| mAPM (Medium) | 55.1% | **60.1%** | +5.0% abs |
| mAPL (Large) | 62.5% | **65.4%** | +2.9% abs |

---

## Computational Efficiency

| Metric | Value |
|:---|:---:|
| Parameters | 2.5 M |
| FLOPs | 257.9 G |
| GPU Throughput | 13.2 FPS |
| Inference Latency | 75.5 ms |

---

## Checkpoint

```python
import torch
checkpoint = torch.load('outputs/cmaf_soem_qfdet_v1.pth', map_location='cpu')
model = CMAPSOEMQFDet(in_channels=256, num_classes=1)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
```

---

## Citation / Acknowledgements

Developed for **Yugma TechFest 2.0 - MedhaDrishti National-Level AI Hackathon**  
Department of Information Science and Engineering, JNNCE Shivamogga.
