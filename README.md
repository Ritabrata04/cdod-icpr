# cdod-icpr
Official implementation of the paper:
Towards Robust Cross-Dataset Object Detection: Generalization under Domain Specificity
Ritabrata Chakraborty, Hrishit Mitra, Shivakumara Palaiahnakote, Umapada Pal
(ICPR 2026)

## Overview

Object detectors often fail when evaluated on datasets different from those they are trained on.  
This project studies **cross-dataset object detection (CD-OD)** using the concept of **setting specificity**:

- **Setting-agnostic datasets:** COCO, Objects365  
- **Setting-specific datasets:** Cityscapes, BDD100K  

We evaluate all train→test pairs and introduce a **CLIP-based open-label evaluation** to separate:

- domain shift effects  
- label taxonomy mismatch  

---

## Datasets

- COCO  
- Objects365  
- Cityscapes  
- BDD100K  

---

## Model

- Faster R-CNN (ResNet-50 FPN)  
- PyTorch + MMDetection  
- Zero-shot cross-dataset transfer (no target data)

---

## Installation

```bash
git clone https://github.com/Ritabrata04/cdod-icpr.git
cd cdod-icpr
pip install -r requirements.txt

