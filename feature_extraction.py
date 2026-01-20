import os
import numpy as np
import pandas as pd
import librosa
import cv2
from pathlib import Path

# ================= CONFIG =================
BASE_DIR = Path(__file__).parent.resolve()
DATASET_DIR = BASE_DIR / "Final_5Class_Dataset"

SPEECH_DIR = DATASET_DIR / "speech"
EGG_DIR = DATASET_DIR / "egg"
METADATA_PATH = DATASET_DIR / "metadata.csv"

OUTPUT_DIR = BASE_DIR / "Extracted_Features"
SPEC_DIR = OUTPUT_DIR / "Spectrograms"

SR = 16000
N_LPC = int(2 + SR / 1000)
N_MFCC = N_LPC

IMG_WIDTH = 292
IMG_HEIGHT = 219

# =========================================

# Create output folders
OUTPUT_DIR.mkdir(exist_ok=True)
SPEC_DIR.mkdir(exist_ok=True)

for class_folder in os.listdir(SPEECH_DIR):
    (SPEC_DIR / "speech" / class_folder).mkdir(parents=True, exist_ok=True)
    (SPEC_DIR / "egg" / class_folder).mkdir(parents=True, exist_ok=True)

# ================= FEATURE FUNCTIONS =================

def extract_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfcc, axis=1), np.std(mfcc, axis=1)

def extract_lpc(y, order):
    try:
        lpc = librosa.lpc(y, order=order)
        return lpc[:order]
    except Exception:
        return np.zeros(order)

def extract_pitch(y, sr):
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    pitch = pitches[mags > np.median(mags)]
    return np.mean(pitch) if len(pitch) > 0 else 0

def extract_jitter(y, sr):
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    pitch = pitches[mags > np.median(mags)]
    if len(pitch) < 2:
        return 0
    return np.mean(np.abs(np.diff(pitch)) / (pitch[:-1] + 1e-6))

def extract_shimmer(y):
    frames = librosa.util.frame(y, frame_length=2048, hop_length=512)
    rms = np.sqrt(np.mean(frames**2, axis=0))
    if len(rms) < 2:
        return 0
    return np.mean(np.abs(np.diff(rms)) / (rms[:-1] + 1e-6))

def extract_hnr(y):
    harmonic, noise = librosa.effects.hpss(y)
    return np.sum(harmonic**2) / (np.sum(noise**2) + 1e-6)

def extract_energy(y):
    return np.mean(librosa.feature.rms(y=y))

def extract_zcr(y):
    return np.mean(librosa.feature.zero_crossing_rate(y))

def extract_spectral_slope(y, sr):
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    return np.polyfit(freqs[:S.shape[0]], np.mean(S, axis=1), 1)[0]

# ================= MEL SPECTROGRAM =================

def save_mel_spectrogram(y, sr, out_path):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=256
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    img = cv2.normalize(mel_db, None, 0, 255, cv2.NORM_MINMAX)
    img = img.astype(np.uint8)
    img_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    img_resized = cv2.resize(img_color, (IMG_WIDTH, IMG_HEIGHT))

    cv2.imwrite(str(out_path), img_resized)

# ================= MAIN PIPELINE =================

metadata = pd.read_csv(METADATA_PATH)

# 🔴 CRITICAL FIX: Check for split column
if 'split' not in metadata.columns:
    print("❌ ERROR: 'split' column missing in metadata.csv!")
    print("Please run data.py first to generate the master split.")
    exit(1)

feature_rows = []

for _, row in metadata.iterrows():
    pid = row["patient_id"]
    label = int(row["label"])
    age = row["age"]
    disease = row["disease"]
    split = row["split"]  # 🔴 PRESERVE SPLIT

    class_folder = f"C{label}_" + disease.replace(" ", "_")

    speech_path = SPEECH_DIR / class_folder / f"{pid}.wav"
    egg_path = EGG_DIR / class_folder / f"{pid}.wav"

    if not speech_path.exists() or not egg_path.exists():
        continue

    # Load audio
    y_speech, _ = librosa.load(speech_path, sr=SR)
    y_egg, _ = librosa.load(egg_path, sr=SR)

    # ===== Handcrafted Features =====
    mfcc_mean, mfcc_std = extract_mfcc(y_speech, SR)
    lpc = extract_lpc(y_speech, N_LPC)
    pitch = extract_pitch(y_speech, SR)
    jitter = extract_jitter(y_speech, SR)
    shimmer = extract_shimmer(y_speech)
    hnr = extract_hnr(y_speech)
    energy = extract_energy(y_speech)
    zcr = extract_zcr(y_speech)
    slope = extract_spectral_slope(y_speech, SR)
    egg_energy = np.mean(y_egg ** 2)

    # ===== Save Mel Spectrograms =====
    save_mel_spectrogram(
        y_speech,
        SR,
        SPEC_DIR / "speech" / class_folder / f"{pid}.png"
    )

    save_mel_spectrogram(
        y_egg,
        SR,
        SPEC_DIR / "egg" / class_folder / f"{pid}.png"
    )

    # ===== Store Features =====
    features = {
        "patient_id": pid,
        "label": label,
        "age": age,
        "split": split,  # 🔴 PRESERVE SPLIT
        "pitch": pitch,
        "jitter": jitter,
        "shimmer": shimmer,
        "hnr": hnr,
        "energy": energy,
        "zcr": zcr,
        "spectral_slope": slope,
        "egg_energy": egg_energy
    }

    for i, val in enumerate(mfcc_mean):
        features[f"mfcc_mean_{i+1}"] = val

    for i, val in enumerate(mfcc_std):
        features[f"mfcc_std_{i+1}"] = val

    for i, val in enumerate(lpc):
        features[f"lpc_{i+1}"] = val

    feature_rows.append(features)

# ================= SAVE CSV =================

df_features = pd.DataFrame(feature_rows)
df_features.to_csv(OUTPUT_DIR / "handcrafted_features.csv", index=False)

print("✅ Feature extraction completed")
print("Total samples:", len(df_features))
print("\nClass distribution:")
print(df_features["label"].value_counts().sort_index())
print("\nSplit distribution:")
print(df_features["split"].value_counts())

# Verify no leakage
train_patients = set(df_features[df_features['split'] == 'train']['patient_id'])
test_patients = set(df_features[df_features['split'] == 'test']['patient_id'])
overlap = train_patients & test_patients

if overlap:
    print(f"\n❌ WARNING: {len(overlap)} patients appear in both train and test!")
else:
    print("\n✅ No patient leakage detected")
