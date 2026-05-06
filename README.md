# Object-Recognition-using-RGB-and-InfraRed-Images-for-Autonomous-Driving

## Overview

This final project for **CSE 547 Deep Learning** explores the challenges and benefits of using both **RGB** and **InfraRed (IR)** imagery for autonomous driving object recognition. The goal is to build, evaluate, and optimize deep learning pipelines to accurately classify objects critical for autonomous driving safety across different imaging modalities.

## Dataset

- **Name:** FLIR ADAS Thermal Dataset v2
- **Classes (8):** `bike`, `bus`, `car`, `person`, `sign`, `motor`, `light`, `truck`

## Project Structure & Methodology

The comprehensive deep learning pipeline is divided into six structured parts:

1. **Part 1 (CNN Architectures):** Built and evaluated baseline Convolutional Neural Networks (CNNs) from scratch to benchmark standard classification capabilities.
2. **Part 2 (Regularization):** Applied regularization strategies (Dropout, L2/Weight Decay, Data Augmentation) to combat overfitting and improve generalization on our core CNNs.
3. **Part 3 (Transfer Learning):** Implemented Transfer Learning explicitly for the RGB modality, fine-tuning pretrained architectures. The strongest performing RGB model leveraged **VGG16 (Freezing Blocks 1 & 2)**.
4. **Part 4 (AutoEncoders):** Trained and optimized Convolutional AutoEncoders exclusively for the IR modality. The latent representations from an optimized Autoencoder (**AE-4**) were extracted and passed to a lightweight, regularized classification head (**clf_reg_light**).
5. **Part 5 (Analysis & Cross-Sensor Comparison):** Conducted rigorous qualitative and quantitative analyses.
    - Analyzed Confusion Matrices and Per-Class F1 comparisons.
    - Tracked "Sensor Advantage", measuring where one modality excels over the other on the same scenarios.
6. **Part 6 (Final Optimization):** Final model retraining and packaging for validation and test predictions.

## Key Findings

- **RGB Model Strengths:** Highly detailed RGB visuals naturally excel at capturing textural details. Pretrained models like **VGG16** transfer heavily contextualized ImageNet features well into this domain, ensuring strong performance for classes like `car`, `bus`, and `sign`.
- **IR Model Strengths:** Infrared imaging natively captures thermal signatures, demonstrating clear comparative robustness in detecting living or mechanical heat sources (e.g., `person`, `motor`). This circumvents typical visual noise like deep shadows, poor lighting, or headlight glares. Using Autoencoder-based feature extraction tailored for IR distribution ultimately outperformed standard deep architectures trained from scratch.
- **Cross-Sensor Need:** Neither sensor provides absolute superiority across every class. The cross-sensor analysis highlights significant sensor blindness per modality; thus, relying on a unified fusion or ensemble mapping between RGB and IR remains critical for optimal autonomous driving computer vision pipelines.
