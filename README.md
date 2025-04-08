# Cassava Leaf Disease Classification

## Overview

This repository contains the code and resources for the Cassava Leaf Disease Classification project, developed for the Applied Machine Learning Systems II (ELEC0135) course (SN: 21008931). The project utilizes deep learning, specifically the EfficientNetB3 model, to classify cassava leaf diseases using images from the Kaggle competition [Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/data).

## Project Structure
```
├── Datasets
│   └── [Place downloaded dataset here]
├── EfficientNetB3.py
├── main.py
├── results
│   └── [Place downloaded Model here and images are saved here]
└── Cassava_Classification.ipynb
```

## Data

Download the dataset from the Kaggle competition [here](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/data). Please place the downloaded dataset in the `Datasets` folder to ensure smooth code execution.

## Notebook and Outputs

- The `Cassava_Classification.ipynb` Jupyter notebook contains the full workflow and detailed output results.
- Output images and confusion matrices generated during the notebook execution are saved in the `results` folder for easy access and review.

## Model and Training

Due to computational limitations of local hardware, model training was conducted on a Kaggle notebook environment. The complete model training pipeline, including Exploratory Data Analysis (EDA), data preprocessing, data augmentation, and training logs, is available on the Kaggle notebook:

[View Kaggle Notebook](https://www.kaggle.com/code/xuanfanlin/cld-classification-xuanfan-lin)

The trained model `Cassava_best_model_stage1.keras` is available for download from the Kaggle notebook and should be placed in the `results` folder.

## Code Execution

The local pipeline consists of the following files:
- `EfficientNetB3.py`: Implements the full training and evaluation pipeline for cassava leaf disease classification using EfficientNetB3
- `main.py`: Runs the pipeline function defined in EfficientNetB3.py as the entry point.

Execute `main.py` to begin the pipeline.

## Required Libraries

Install the following Python packages before running the code:

```bash
pip install numpy==1.26.4 pandas==2.2.3 matplotlib==3.10.0 seaborn==0.13.2 tensorflow==2.16.2 scikit-learn

```

The specific libraries used in this project:

```python
import os
import random
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
```



