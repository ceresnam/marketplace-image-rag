# Marketplace Image Classification

## Project Overview
This project implements an image classification system for marketplace product images using transfer learning on EfficientNetV2. The system automatically categorizes product images into 150 marketplace categories based on uploaded photos.

### Business Context
- **150 product categories** with thousands of existing products per category
- **Goal:** Automatically classify product photos into the correct category
- **Challenge:** Handle diverse real-world image quality from user smartphone uploads (varying lighting, angles, backgrounds)

## Features
- Image preprocessing and normalization
- Multi-class product classification
- Category prediction
- Quality assessment

## Tech Stack

### Core Framework
- **Model:** EfficientNetV2-S (Transfer Learning)
- **Framework:** PyTorch / PyTorch Lightning (Advanced)
- **Image Size:** 384x384 pixels (EfficientNetV2-S standard)
- **Deployment:** FastAPI on CPU (Standard) / ONNX Runtime/TensorRT on GPU (High Scale)

### Infrastructure
- Python 3.x
- **uv** for fast package management
- Databricks for distributed computing and MLflow tracking
- Apache Spark for data processing
- Image processing libraries (Pillow, OpenCV)

### Why EfficientNetV2-S with PyTorch?
- **Modern architecture** - Faster training than original EfficientNet (up to 5-11x speedup)
- **Excellent accuracy** (~84.9% Top-1 accuracy on ImageNet)
- **Efficient training** - Uses Fused-MBConv blocks for better speed/accuracy trade-off
- **Better parameter efficiency** (~24M parameters)
- **Optimized for transfer learning** - Progressive learning during training
- **PyTorch ecosystem** provides excellent pre-trained models via `torchvision` and `timm` libraries

### Why Better Than AlexNet, ResNet, and Original EfficientNet?

| Model | Year | Parameters | Accuracy | Training Speed | Why Inferior |
|-------|------|-----------|----------|----------------|--------------|
| **AlexNet** | 2012 | ~61M | ~56.5% | Fast | ❌ Outdated architecture, poor accuracy, inefficient |
| **ResNet-50** | 2015 | ~25M | ~76.1% | Medium | ⚠️ Good but slower training, lower accuracy |
| **ResNet-101** | 2015 | ~44M | ~77.4% | Slow | ⚠️ Heavyweight, diminishing returns |
| **EfficientNet-B0** | 2019 | ~5.3M | ~77.1% | Medium | ⚠️ Good accuracy but slower training |
| **EfficientNet-B4** | 2019 | ~19M | ~82.9% | Slow | ⚠️ High accuracy but very slow training |
| **EfficientNetV2-S** | 2021 | ~24M | ~84.9% | Fast | ✅ Best accuracy-to-efficiency ratio |



**Bottom Line:** For 150 marketplace categories, you need a model that captures fine-grained details (e.g., distinguishing "Leather Jacket" from "Faux Leather Jacket"). AlexNet is too primitive, ResNet is good but outdated, original EfficientNet is accurate but slow to train, and EfficientNetV2-S is the modern standard with the best speed-accuracy trade-off.

## Dataset Requirements

### Production-Ready Target (Recommended)
- **500-1,000 images per category**
- **Total Dataset:** ~75,000 - 150,000 images
- **Source:** Real user-uploaded photos from existing marketplace products

### Dataset Quality Guidelines
| Scenario | Verdict |
|----------|---------|
| 1,000 Exact Duplicates | ❌ Useless - Effective size is 1 |
| 1,000 Nearly Identical Photos (same angle/lighting) | ❌ Bad - No diversity for model to generalize |
| 500-1,000 Diverse User Uploads | ✅ Excellent - Variety in lighting, angles, backgrounds trains robust model |

### Training Strategy
1. **Start with 1,000 images per category** (randomly sampled from existing products)
2. Train baseline EfficientNet-B0 model
3. **Analyze confusion matrix** to identify problem categories
4. Scale to 2,000-5,000 per category only if needed

### Handling "Out of Distribution" (OOD)
**Problem:** Marketplace is an "open world" - users may upload items outside 150 categories (cars, pets, prohibited items)

**Solution:** Implement OOD detector or "Trash" category trained on random irrelevant images to handle "I don't know" cases

## Model Architecture

### EfficientNetV2 Model Variants
| Model | Input Size | Parameters | Accuracy | Training VRAM (batch=32) | Inference VRAM | Use Case |
|-------|-----------|------------|----------|--------------------------|----------------|----------|
| **EfficientNetV2-S** | **384x384** | **~24M** | **~84.9%** | **~8-10 GB** | **~2 GB** | **✅ Recommended** |
| EfficientNetV2-M | 480x480 | ~54M | ~85.7% | ~16-20 GB | ~4 GB | High-accuracy needs |
| EfficientNetV2-L | 480x480 | ~119M | ~86.3% | ~24-32 GB | ~6 GB | Research/Max accuracy |

**Note:** VRAM requirements vary based on batch size, mixed precision (FP16/FP32), and framework optimizations. Values shown are approximate for full precision (FP32) training. Using mixed precision (FP16) can reduce VRAM by ~40-50%.

**Architecture:** Transfer Learning approach
- Start with pre-trained EfficientNetV2-S
- Fine-tune on marketplace product dataset
- Replace final classification layer for 150 categories + OOD class
- Leverage progressive learning strategy during fine-tuning

## Training

### Transfer Learning Approach
| Approach | Images/Category | Computational Cost | Status |
|----------|----------------|-------------------|--------|
| Feature Extraction | 100-500 | Low | ✅ Sufficient data available |
| **Fine-Tuning** | **500-2,000** | Medium | **✅ Recommended approach** |
| Training from Scratch | 5,000+ | Very High | ❌ Unnecessary & wasteful |

### Training Pipeline (Databricks)
1. **Data Preparation**
   - Load images from existing marketplace products
   - Resize images to 384x384 pixels
   - Split: 70% train, 15% validation, 15% test
   - Apply data augmentation (rotation, flip, brightness, contrast, cutout)

2. **Model Training**
   - Load pre-trained EfficientNetV2-S with ImageNet weights
   - Freeze early layers (feature extraction)
   - Fine-tune later layers + new classification head
   - Use progressive learning (gradually increase image size during training)
   - Use MLflow for experiment tracking
   - Leverage distributed training across cluster nodes

3. **Model Evaluation**
   - Analyze confusion matrix for misclassified categories
   - Identify visually similar categories that need more data
   - Test robustness across diverse user upload conditions (lighting, angles, backgrounds)

4. **Model Registry**
   - Store best models in MLflow Model Registry
   - Version control for production deployments

## Evaluation

### Key Metrics
- **Top-1 Accuracy:** Percentage of correct first predictions
- **Top-5 Accuracy:** Correct category in top 5 predictions
- **Confusion Matrix:** Identify commonly confused categories
- **Per-Category Performance:** Find underperforming categories

### Trust vs. Speed Trade-off
**Marketplace Priority: Trust > Speed**
- Misclassifying a "Gucci Bag" as "Luggage" breaks user trust
- EfficientNetV2-S's high accuracy (~85%) is critical for user trust

## Deployment Strategy

### Standard Deployment
- FastAPI REST API on CPU
- Docker containerized
- Horizontal scaling for load balancing

### High-Scale Deployment
- ONNX Runtime or TensorRT on GPU
- Batch inference optimization

## Future Improvements & Scaling Path

### Model Upgrades
- **If V2-S accuracy insufficient:** Upgrade to EfficientNetV2-M or V2-L
  - V2-M: 480x480 input, ~85.7% accuracy
  - V2-L: 480x480 input, ~86.3% accuracy (max performance)
- **Advanced alternative:** Vision Transformer (ViT) or CLIP for zero-shot classification

### Feature Enhancements
- Multi-scale image processing and ensemble
- Ensemble models for critical/high-value categories
- Active learning pipeline for continuous improvement
- A/B testing framework for model updates
