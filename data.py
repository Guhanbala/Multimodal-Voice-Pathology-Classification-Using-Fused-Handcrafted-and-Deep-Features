import os
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split

# ================= USER CONFIGURATION =================
DATASET_FOLDER_NAME = "dataset"
BASE_DIR = Path(__file__).parent.resolve()
ROOT_DATA_FOLDER = BASE_DIR / DATASET_FOLDER_NAME
OUTPUT_FOLDER = BASE_DIR / "Final_5Class_Dataset"
RANDOM_STATE = 42  # Fixed seed for reproducibility

# ================= CLASS LABEL MAPPING =================
CLASS_MAP = {
    "healthy": 0,
    "Laryngitis": 1,
    "Hyperfunktionelle Dysphonie": 2,
    "Kontaktpachydermie": 3,
    "Rekurrensparese": 4
}

CLASS_NAME_MAP = {
    0: "C0_healthy",
    1: "C1_Laryngitis",
    2: "C2_Hyperfunktionelle_Dysphonie",
    3: "C3_Kontaktpachydermie",
    4: "C4_Rekurrensparese"
}

def convert_to_wav(input_path, output_path):
    try:
        cmd = ['ffmpeg', '-y', '-i', str(input_path), str(output_path), '-v', 'error']
        subprocess.run(cmd, check=True)
        return True
    except Exception:
        return False

def calculate_age(birth_date, recording_date):
    try:
        birth = datetime.strptime(str(birth_date).strip(), '%Y-%m-%d')
        rec = datetime.strptime(str(recording_date).strip(), '%Y-%m-%d')
        return rec.year - birth.year - ((rec.month, rec.day) < (birth.month, birth.day))
    except Exception:
        return None

def get_ages_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path, encoding='latin-1')
        if 'AufnahmeID' not in df.columns:
            df = pd.read_csv(csv_path, sep=';', encoding='latin-1')
        patient_ages = {}
        for _, row in df.iterrows():
            pid = str(row['AufnahmeID'])
            if pd.notna(row['Geburtsdatum']) and pd.notna(row['AufnahmeDatum']):
                age = calculate_age(row['Geburtsdatum'], row['AufnahmeDatum'])
                if age is not None:
                    patient_ages[pid] = age
        return patient_ages
    except Exception:
        return {}

def process_disease_folder(disease_folder):
    disease_name = disease_folder.name
    if disease_name not in CLASS_MAP:
        return []

    label = CLASS_MAP[disease_name]
    class_folder = CLASS_NAME_MAP[label]
    csv_path = disease_folder / "overview.csv"

    if not csv_path.exists():
        return []

    print(f"[*] Processing {disease_name} (Label {label})")
    patient_ages = get_ages_from_csv(csv_path)
    metadata = []

    for root, _, files in os.walk(disease_folder):
        for file in files:
            if "a_n" in file and "phrase" not in file and "egg" not in file.lower():
                pid = file.split("-")[0]
                if pid not in patient_ages:
                    continue

                vowel_path = Path(root) / file
                egg_path = None
                for f in os.listdir(root):
                    if pid in f and "egg" in f.lower() and "a_n" in f:
                        egg_path = Path(root) / f
                        break

                if egg_path is None:
                    continue

                wav_name = f"{pid}.wav"
                if convert_to_wav(vowel_path, DEST_VOWELS / class_folder / wav_name):
                    convert_to_wav(egg_path, DEST_EGG / class_folder / wav_name)
                    metadata.append({
                        "patient_id": pid,
                        "age": patient_ages[pid],
                        "disease": disease_name,
                        "label": label
                    })
    return metadata

def main():
    global DEST_VOWELS, DEST_EGG
    DEST_VOWELS = OUTPUT_FOLDER / "speech"
    DEST_EGG = OUTPUT_FOLDER / "egg"

    DEST_VOWELS.mkdir(parents=True, exist_ok=True)
    DEST_EGG.mkdir(parents=True, exist_ok=True)

    for _, class_name in CLASS_NAME_MAP.items():
        (DEST_VOWELS / class_name).mkdir(parents=True, exist_ok=True)
        (DEST_EGG / class_name).mkdir(parents=True, exist_ok=True)

    all_metadata = []
    for disease_dir in ROOT_DATA_FOLDER.iterdir():
        if disease_dir.is_dir():
            all_metadata.extend(process_disease_folder(disease_dir))

    if not all_metadata:
        print("[!] No data processed.")
        return

    df = pd.DataFrame(all_metadata)
    df = df.drop_duplicates(subset=["patient_id"])

    # ================= 🚨 CRITICAL ADDITION: MASTER SPLIT =================
    print("\n🔹 Creating Master Train/Test Split...")
    # Stratify by label to ensure balanced classes in both sets
    train_ids, test_ids = train_test_split(
        df["patient_id"], 
        test_size=0.2, 
        stratify=df["label"], 
        random_state=RANDOM_STATE
    )
    
    # Map back to dataframe
    df["split"] = "train"
    df.loc[df["patient_id"].isin(test_ids), "split"] = "test"
    # ======================================================================

    df.to_csv(OUTPUT_FOLDER / "metadata.csv", index=False)

    print("\n[+] Dataset preparation complete.")
    print("\nClass distribution:")
    print(df['label'].value_counts())
    print("\nSplit distribution:")
    print(df['split'].value_counts())

if __name__ == "__main__":
    main()