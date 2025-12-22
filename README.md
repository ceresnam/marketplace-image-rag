# Marketplace Image Classification

## Project Overview

This project implements an image RAG system for marketplace product images using transfer learning on the DinoV2 vision transformer model. The system automatically categorizes product images into 150 marketplace categories based on user-uploaded photos and searches for photos of similar products.

### Business Context

- **150 product categories** with thousands of existing products per category
- **Goal:** Automatically classify product photos into the correct category. Search similar product photos.
- **Challenge:** Handle diverse real-world image quality from user smartphone uploads (varying lighting, angles, backgrounds)

## Features

- Image preprocessing and normalization
- Multi-class product classification
- Category prediction

## Tech Stack

### Core Framework

- **Model:** DinoV2 (Transfer Learning)
- **Framework:** PyTorch Lightning
- **Image Size:** 224x224 pixels (DinoV2 standard)

### Infrastructure

- Python 3.12
- uv for package management
- Databricks Apache Spark for dataset preprocessing
- Image processing libraries (Pillow)
- PostgreSQL for vector database

## Project Folder Structure

```
marketplace-image-rag/
├── README.md                        # This file
├── pyproject.toml                   # Project dependencies (uv)
│
├── marketplace/                         # Dataset and model code
│   ├── const.py                     # Constants (ImageNet stats, etc.)
│   ├── dataset.py                   # PyTorch Dataset and DataModule
│   ├── db.py                        # Database utilities
│   ├── loss.py                      # Loss functions
│   └── model.py                     # DinoV2Classification model
│
├── notebooks/                       # Jupyter notebooks
│   ├── 01 explore.ipynb             # Data exploration
│   ├── 02 embed.ipynb               # Embedding generation
│   ├── 03 similarity search.ipynb   # Similarity search
│   ├── 04 classification.ipynb      # Model training
│   └── 05 category prediction.ipynb # Inference and evaluation
│
├── etl/                             # ETL pipeline
│
├── checkpoints/                     # Model
│
├── scripts/                         # Shell scripts
│   └── devenv.sh                    # Development environment setup
│
├── config/                          # Configuration files
│   ├── expose.yaml                  # Service exposure config
│   ├── postgres.yaml                # PostgreSQL configuration
│   └── spark-values.yaml            # Spark configuration
│
├── docs/                            # Documentation and design choices
│   ├── model.md
│   ├── loss function.md
│   └── text to image search.md
│
└── data -> /mnt/data                # Symlink to data directory
```
