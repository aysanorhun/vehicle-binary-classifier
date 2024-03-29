# Car vs Bike Image Classification

## Overview
This project explores the effectiveness of various machine learning approaches for distinguishing between car and motorcycle images. The project benchmarks four different algorithms with hyperparameter tuning and data augmentation.

**Best Performance:** CNN with data augmentation achieved **89.88% accuracy**

## Key Features
- **Multiple Model Implementations**: KNN, Random Forest, SVM, and CNN
- **Data Augmentation**: Horizontal flips, Gaussian blur, and noise injection
- **Dimensionality Reduction**: PCA analysis with varying component counts
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, and F1-Score metrics
- **Hyperparameter Optimization**: Systematic tuning for each algorithm
- **Comparative Analysis**: Detailed performance comparison across all models

## Dataset
- **Source**: [Car vs Bike Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/utkarshsaxenadn/car-vs-bike-classification-dataset)
- **Size**: 4,000 images (2,000 cars + 2,000 motorcycles)
- **Preprocessing**:
  - Resized to 64×64 pixels
  - Converted to grayscale
  - Flattened for classical ML algorithms
- **Split**: 60% training, 20% validation, 20% testing

## Models & Results
| Model                             | Configuration        | Test Accuracy | Recall | Precision | F1-Score |
|-----------------------------------|----------------------|---------------|--------|-----------|----------|
| **CNN** (Augmented)               | 4-layer architecture | **89.88%**    | 0.930  | 0.935     | 0.933    |
| **CNN** (Non-augmented)           | 4-layer architecture | **88.75%**    | 0.925  | 0.940     | 0.932    |
| **SVM** (Augmented)               | RBF kernel           | **83.50%**    | 0.840  | 0.840     | 0.840    |
| **SVM** (Non-augmented)           | RBF kernel           | **83.13%**    | 0.825  | 0.825     | 0.824    |
| **Random Forest** (Non-augmented) | 50 estimators        | **81.13%**    | 0.841  | 0.841     | 0.841    |
| **Random Forest** (Augmented)     | 100 estimators       | **80.25%**    | 0.817  | 0.817     | 0.817    |
| **KNN** (Non-augmented)           | k=5                  | **70.75%**    | 0.729  | 0.731     | 0.727    |
| **KNN** (Augmented)               | k=9                  | **69.37%**    | 0.730  | 0.730     | 0.727    |

### Best Model Architecture (CNN)
```
Input (64×64×1)
  ↓
Conv2D (32 filters, 3×3) + ReLU
  ↓
MaxPooling2D (2×2)
  ↓
Conv2D (64 filters, 3×3) + ReLU
  ↓
MaxPooling2D (2×2)
  ↓
Conv2D (128 filters, 3×3) + ReLU
  ↓
MaxPooling2D (2×2)
  ↓
Flatten
  ↓
Dense (128 units) + ReLU
  ↓
Dense (1 unit) + Sigmoid
```

## Project Structure
```
CS464_Project/
├── code.ipynb              # Main Jupyter notebook with all experiments
├── docs/
│   └── report.pdf          # Comprehensive project report (26 pages)
├── Car-Bike-Dataset/       # Dataset directory (not included in repo)
│   ├── Car/
│   └── Bike/
├── pyproject.toml
├── .gitignore
└── README.md
```

## Methodology

### 1. Data Preprocessing
- Image resizing to 64×64 pixels
- Grayscale conversion
- Normalization
- Train-validation-test split (60-20-20)

### 2. Data Augmentation
Applied transformations:
- Horizontal flips (50% probability)
- Gaussian blur (σ ∈ [0, 0.5])
- Additive Gaussian noise
- Random order application

### 3. Hyperparameter Tuning
**KNN**: Tested k values from 5 to 37 (step=2)
- Best non-augmented: k=5
- Best augmented: k=9

**Random Forest**: Tested estimators [10, 20, 50, 100, 150]
- Best both: n_estimators=50

**SVM**: Tested kernels [linear, poly, rbf, sigmoid]
- Best both: RBF kernel

**CNN**: Tested 4 different architectures
- Best: 4-layer architecture with additional dense layer

### 4. PCA Analysis
Tested component counts: [1, 50, 100, 250, 500, 1000, 2400]
- Optimal for KNN/SVM: 50-100 components
- Found diminishing returns beyond 100 components
