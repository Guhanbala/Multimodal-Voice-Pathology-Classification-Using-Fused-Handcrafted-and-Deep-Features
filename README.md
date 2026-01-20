# 🎙️ Hierarchical Voice Pathology Detection System

A deep learning-based system for detecting and classifying voice pathologies using speech signals and electroglottography (EGG) data. The system employs a two-stage hierarchical approach: first detecting healthy vs. pathological voices, then classifying the type of pathology.

## 📋 Overview

This project implements a comprehensive pipeline for voice pathology analysis:

- **Stage 1 (Detection)**: Binary classification (Healthy vs. Pathological)
- **Stage 2 (Diagnosis)**: Multi-class classification of pathology types
  - Structural/Inflammatory (Laryngitis + Kontaktpachydermie)
  - Hyperfunctional (Hyperfunktionelle Dysphonie)
  - Neurological (Rekurrensparese)

## 🏗️ System Architecture

### Feature Extraction
- **Handcrafted Features**: MFCC, LPC, pitch, jitter, shimmer, HNR, energy, ZCR, spectral slope
- **Deep Features**: Fine-tuned ResNet18 on mel spectrograms (both speech and EGG signals)
- **Multi-modal Fusion**: Combination of handcrafted + speech CNN + EGG CNN features

### Classification Model
- Two-stage hierarchical MLP (Multi-Layer Perceptron)
- Patient-level 80/20 train/test split to prevent data leakage
- Early stopping with validation monitoring

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- FFmpeg (for audio conversion)
- CUDA-capable GPU (optional, but recommended for ResNet training)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/voice-pathology-detection.git
cd voice-pathology-detection
```

2. **Create virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install FFmpeg**
- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## 📂 Project Structure

```
voice-pathology-detection/
├── dataset/                          # Raw dataset folder (not included)
│   ├── healthy/
│   ├── Laryngitis/
│   ├── Hyperfunktionelle Dysphonie/
│   ├── Kontaktpachydermie/
│   └── Rekurrensparese/
├── data.py                          # Dataset preparation & WAV conversion
├── feature_extraction.py            # Handcrafted features & spectrograms
├── resnet18_deep_features.py       # Deep feature extraction with ResNet18
├── fuse_data.py                    # Feature fusion & label grouping
├── final.py                        # Hierarchical MLP training & evaluation
├── requirements.txt
└── README.md
```

## 🔄 Pipeline Execution

Run the scripts in this order:

### 1. Data Preparation
```bash
python data.py
```
- Converts audio files to WAV format
- Extracts patient metadata (age, disease)
- Creates 80/20 train/test split at patient level
- Outputs: `Final_5Class_Dataset/`

### 2. Feature Extraction
```bash
python feature_extraction.py
```
- Extracts handcrafted audio features
- Generates mel spectrograms for speech and EGG signals
- Outputs: `Extracted_Features/`

### 3. Deep Feature Extraction
```bash
python resnet18_deep_features.py
```
- Fine-tunes ResNet18 on mel spectrograms
- Extracts deep features from penultimate layer
- Trains separate models for speech and EGG
- Outputs: `Deep_Features/`

### 4. Feature Fusion
```bash
python fuse_data.py
```
- Merges handcrafted + speech CNN + EGG CNN features
- Groups labels (1 & 3 → Structural/Inflammatory)
- Outputs: `fused_dataset_4class.csv`

### 5. Hierarchical Training & Evaluation
```bash
python final.py
```
- Trains two-stage hierarchical classifier
- Evaluates on three levels:
  - Level 1: Detection (Healthy vs. Sick)
  - Level 2: Diagnosis (Pathology type)
  - Overall: End-to-end system accuracy
- Generates confusion matrices

## 📊 Dataset Format

Place your dataset in a `dataset/` folder with this structure:

```
dataset/
├── healthy/
│   ├── overview.csv
│   └── [patient audio files]
├── Laryngitis/
│   ├── overview.csv
│   └── [patient audio files]
└── ...
```

Each `overview.csv` should contain:
- `AufnahmeID`: Patient ID
- `Geburtsdatum`: Birth date (YYYY-MM-DD)
- `AufnahmeDatum`: Recording date (YYYY-MM-DD)

## 🎯 Class Mapping

| Original Label | Disease | Grouped Label | Class Name |
|---------------|---------|---------------|------------|
| 0 | Healthy | 0 | Healthy |
| 1 | Laryngitis | 1 | Structural/Inflammatory |
| 2 | Hyperfunktionelle Dysphonie | 2 | Hyperfunctional |
| 3 | Kontaktpachydermie | 1 | Structural/Inflammatory |
| 4 | Rekurrensparese | 3 | Neurological |

## 📈 Results

The system outputs three confusion matrices:

- `cm_level1_detection.png` - Binary detection performance
- `cm_level2_diagnosis.png` - Pathology classification (on sick patients only)
- `cm_overall.png` - End-to-end system performance

## ⚙️ Configuration

Key parameters can be modified in each script:

**data.py**
- `RANDOM_STATE = 42` - Reproducibility seed
- `DATASET_FOLDER_NAME = "dataset"` - Input folder

**feature_extraction.py**
- `SR = 16000` - Sampling rate
- `N_MFCC` - Number of MFCC coefficients

**resnet18_deep_features.py**
- `EPOCHS = 7` - Fine-tuning epochs
- `BATCH_SIZE = 16` - Training batch size
- `NUM_CLASSES = 4` - After grouping

**final.py**
- `EPOCHS = 100` - MLP training epochs
- `LR = 0.001` - Learning rate
- `PATIENCE = 20` - Early stopping patience

