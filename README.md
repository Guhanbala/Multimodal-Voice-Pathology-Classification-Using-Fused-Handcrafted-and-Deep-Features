# 🎙️ Hierarchical Voice Pathology Detection System

A deep learning-based system for detecting and classifying voice pathologies using speech signals and electroglottography (EGG) data. The system employs a two-stage hierarchical approach: first detecting healthy vs. pathological voices, then classifying the type of pathology.

## 📋 Overview

This project implements a comprehensive pipeline for voice pathology analysis:

- **Stage 1 (Detection)**: Binary classification (Healthy vs. Pathological)
- **Stage 2 (Diagnosis)**: Multi-class classification of 3 pathology types
  - Structural/Inflammatory (Laryngitis + Kontaktpachydermie)
  - Hyperfunctional (Hyperfunktionelle Dysphonie)
  - Neurological (Rekurrensparese)

## 🏗️ System Architecture

### Feature Extraction
- **Handcrafted Features**: MFCC (mean & std), LPC, pitch, jitter, shimmer, HNR, energy, ZCR, spectral slope, and EGG energy
- **Deep Features**: ResNet18 feature extraction on mel spectrograms (both speech and EGG signals)
- **Multi-modal Fusion**: Combination of handcrafted + speech CNN + EGG CNN features

### Classification Model
- Two-stage hierarchical MLP (Multi-Layer Perceptron)
  - Stage 1: 64 → 32 → 2 (Binary: Healthy/Pathological)
  - Stage 2: 64 → 32 → 3 (Diagnosis: 3 pathology types)
- Patient-level 80/20 train/test split to prevent data leakage
- Early stopping with F1-score monitoring and patience=20
- Batch normalization and dropout (0.5) for regularization

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- FFmpeg (for audio conversion)
- CUDA-capable GPU (optional, but recommended)

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

## 📂 Project Structure

```
voice-pathology-detection/
├── dataset/                          # Raw dataset folder (not included)
│   ├── healthy/
│   │   ├── overview.csv
│   │   └── [patient audio files]
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

**What it does:**
- Converts all audio files to WAV format (16kHz, mono) using FFmpeg
- Extracts patient metadata (patient ID, age, disease label)
- Pairs speech audio with corresponding EGG signals
- Creates stratified 80/20 train/test split at patient level to prevent data leakage
- Saves split information in metadata for downstream use

**Outputs:**
- `Final_5Class_Dataset/speech/` - Converted speech WAV files organized by class
- `Final_5Class_Dataset/egg/` - Converted EGG WAV files organized by class
- `Final_5Class_Dataset/metadata.csv` - Patient metadata with train/test split

### 2. Feature Extraction
```bash
python feature_extraction.py
```

**What it does:**
- Extracts handcrafted audio features from speech signals:
  - MFCC: Mean and standard deviation of 18 coefficients
  - LPC: Linear Prediction Coefficients (18 values)
  - Pitch, jitter, shimmer, HNR, energy, ZCR, spectral slope
  - EGG energy from paired EGG signals
- Generates 292×219 mel spectrograms (128 mels, Jet colormap) for both speech and EGG
- Preserves train/test split from data.py to prevent leakage
- Verifies no patient overlap between train and test sets

**Outputs:**
- `Extracted_Features/handcrafted_features.csv` - All handcrafted features with split labels
- `Extracted_Features/Spectrograms/speech/` - Speech mel spectrograms (PNG)
- `Extracted_Features/Spectrograms/egg/` - EGG mel spectrograms (PNG)

### 3. Deep Feature Extraction
```bash
python resnet18_deep_features.py
```

**What it does:**
- Loads pretrained ResNet18 (ImageNet weights)
- Removes classification head to extract 512-dimensional feature vectors
- Processes mel spectrograms (speech and EGG separately)
- Extracts features from penultimate layer for both modalities

**Outputs:**
- `Deep_Features/resnet18_speech_features.csv` - 512 ResNet features per speech spectrogram
- `Deep_Features/resnet18_egg_features.csv` - 512 ResNet features per EGG spectrogram

### 4. Feature Fusion
```bash
python fuse_data.py
```

**What it does:**
- Merges three feature sources: handcrafted + speech CNN + EGG CNN
- Groups original 5 classes into 4-class problem:
  - 0 → 0 (Healthy)
  - 1 → 1 (Laryngitis → Structural/Inflammatory)
  - 2 → 2 (Hyperfunctional)
  - 3 → 1 (Kontaktpachydermie → Structural/Inflammatory)
  - 4 → 3 (Neurological)
- Handles missing values (filled with 0)
- Preserves patient_id and split for training pipeline

**Outputs:**
- `fused_dataset_4class.csv` - Combined features with grouped labels (ready for training)

### 5. Hierarchical Training & Evaluation
```bash
python final.py
```

**What it does:**
- Loads fused dataset and auto-generates split if missing
- Standardizes features using StandardScaler
- **Stage 1 Training**: Binary classifier (Healthy vs. Pathological)
  - Trained on 80% of data with 20% validation split
  - Early stopping based on macro F1-score
- **Stage 2 Training**: Multi-class classifier (3 pathology types)
  - Trained only on pathological samples
  - Maps labels: 1→0, 2→1, 3→2 for 3-class problem
- **Evaluation**: Reports three levels of accuracy:
  - **Level 1**: Detection performance (binary)
  - **Level 2**: Diagnosis performance (only on sick patients)
  - **Overall**: End-to-end hierarchical system accuracy

**Outputs:**
- `cm_level1_detection.png` - Confusion matrix: Healthy vs. Sick detection
- `cm_level2_diagnosis.png` - Confusion matrix: Pathology type classification
- `cm_overall.png` - Confusion matrix: Overall system performance
- Console output with accuracy metrics for all three levels

## 📊 Dataset Format

Place your dataset in a `dataset/` folder with this structure:

```
dataset/
├── healthy/
│   ├── overview.csv
│   └── [patient audio files: {patientID}-{recording}-a_n.{ext}, etc.]
├── Laryngitis/
│   ├── overview.csv
│   └── [patient audio files]
├── Hyperfunktionelle Dysphonie/
├── Kontaktpachydermie/
└── Rekurrensparese/
```

### CSV Format (overview.csv)

Each `overview.csv` must contain:
- `AufnahmeID`: Unique patient identifier (string)
- `Geburtsdatum`: Birth date in YYYY-MM-DD format
- `AufnahmeDatum`: Recording date in YYYY-MM-DD format

**Example:**
```
AufnahmeID,Geburtsdatum,AufnahmeDatum
001-001,1980-05-15,2022-03-20
002-001,1975-11-03,2022-04-10
```

### Audio Files

For each patient, the system expects:
- A vowel/speech file containing "a_n" in filename and not containing "phrase" or "egg"
- A corresponding EGG file containing both the patient ID and "egg" in filename

Example filenames:
- Speech: `001-a_n_001.wav` or `001-recording_a_n.mp3`
- EGG: `001-a_n_egg.wav` or `001-recording_a_n_egg.mp3`

## 🎯 Class Mapping

| Original | Disease | Final | Description |
|----------|---------|-------|-------------|
| 0 | Healthy | 0 | Healthy voice |
| 1 | Laryngitis | 1 | Structural/Inflammatory |
| 2 | Hyperfunktionelle Dysphonie | 2 | Hyperfunctional |
| 3 | Kontaktpachydermie | 1 | Structural/Inflammatory |
| 4 | Rekurrensparese | 3 | Neurological |

## 📈 Results

The system produces three confusion matrices showing performance at different levels:

- **Level 1 (Detection)**: Binary classification accuracy for healthy vs. pathological
- **Level 2 (Diagnosis)**: Multi-class accuracy for pathology type (evaluated only on pathological samples)
- **Overall**: Hierarchical system accuracy combining both stages

## ⚙️ Configuration

Key parameters can be modified in each script:

### data.py
- `RANDOM_STATE = 42` - Reproducibility seed for train/test split
- `DATASET_FOLDER_NAME = "dataset"` - Input folder name

### feature_extraction.py
- `SR = 16000` - Sampling rate (Hz)
- `N_LPC = int(2 + SR / 1000)` - LPC order (18 for 16kHz)
- `N_MFCC = N_LPC` - Number of MFCC coefficients (18)
- `IMG_WIDTH = 292, IMG_HEIGHT = 219` - Spectrogram dimensions

### resnet18_deep_features.py
- `DEVICE` - Auto-detects CUDA; set to "cpu" to force CPU
- `FEATURE_DIM = 512` - ResNet18 output dimension

### final.py
- `EPOCHS = 100` - Maximum training epochs per stage
- `LR = 0.001` - Adam optimizer learning rate
- `WEIGHT_DECAY = 1e-4` - L2 regularization
- `PATIENCE = 20` - Early stopping patience
- `DEVICE = "cpu"` - Set to "cuda" for GPU acceleration

## 🔐 Data Leakage Prevention

- **Patient-level splits**: Train/test split is done at patient level, not sample level
- **Consistent splits**: All downstream scripts preserve the split from data.py
- **Verification**: feature_extraction.py explicitly checks for patient overlap
- **Final validation**: final.py confirms no test patients appear in training data

## 📦 Requirements

Key dependencies:
- `torch` - Deep learning framework
- `torchvision` - ResNet18 and image transforms
- `librosa` - Audio feature extraction
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Preprocessing and metrics
- `opencv-python` - Image processing for spectrograms
- `matplotlib` - Visualization
- `tqdm` - Progress bars

See `requirements.txt` for complete list with versions.

