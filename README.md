
# MarsDynSeg / MarTA-Net: Attention-Based Semantic Segmentation for Martian Terrain Analysis

Reliable and automated segmentation of Martian terrain is critical for planetary exploration, rover navigation, geological analysis, and landing-site selection. However, the Martian surface presents several challenges, including highly repetitive textures, subtle terrain variations, complex geological structures, limited annotated datasets, and severe class imbalance among terrain categories. These limitations reduce the effectiveness of conventional semantic segmentation models in identifying safe and scientifically relevant rover landing regions.

This repository presents **MarsDynSeg (MarTA-Net)**, a dynamic attention-based semantic segmentation framework built upon the U-Net architecture and enhanced using the **Convolutional Block Attention Module (CBAM)**. By integrating sequential channel and spatial attention mechanisms, the proposed model refines intermediate feature representations and enables the network to focus on both **what** and **where** important terrain features are located within Martian orbital imagery.

The framework is designed for adaptive rover landing surface prediction and large-scale Martian terrain understanding. The model categorizes terrain into multiple geological classes, including:

- Extremely Smooth Terrain  
- Smooth Terrain  
- Rough Terrain  
- Extremely Rough Terrain  
- Craters  
- Alluvial Fans  

To address the challenge of insufficient labeled data, a **self-updating retraining framework** is incorporated to continuously improve segmentation performance using iterative pseudo-label refinement. Additionally, to mitigate class imbalance — especially for rare but scientifically significant classes such as alluvial fans — a combined loss function consisting of **Weighted Cross-Entropy Loss** and **Dice Loss** is employed.

The model is trained and evaluated using high-resolution Martian orbital imagery obtained from the **Mars Orbital Data Explorer (ODE)**, including **CTX** and **HiRISE** datasets. Generalization capability is further evaluated using the **S5Mars dataset**. Extensive experiments compare the proposed architecture against several state-of-the-art segmentation models, including:

- U-Net  
- DeepLabV3+  
- DPT  
- SegFormer  

Performance evaluation is conducted using standard semantic segmentation metrics such as:

- Intersection over Union (IoU)  
- Dice Coefficient  
- Precision  
- Recall  

Experimental results demonstrate that the proposed attention-enhanced architecture achieves improved segmentation accuracy and more precise Martian terrain mapping compared to baseline approaches. Furthermore, to support deployment in computationally constrained environments, a **quantized lightweight student model** is developed for efficient onboard inference and edge-device deployment in future autonomous planetary missions.

## Repository Contents

This repository contains:

- Model architectures  
- Training and evaluation pipelines  
- Dataset preprocessing scripts  
- Attention-based U-Net implementations  
- Quantization workflows  
- Experimental notebooks  
- Benchmarking and comparison code  

## Objective

The project aims to contribute toward robust AI-driven planetary surface analysis and safer autonomous exploration systems for future Mars missions.

