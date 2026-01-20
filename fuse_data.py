import pandas as pd
from pathlib import Path

# ================= CONFIG =================
BASE_DIR = Path(__file__).parent.resolve()

HANDCRAFTED_CSV = BASE_DIR / "Extracted_Features" / "handcrafted_features.csv"
SPEECH_CNN_CSV  = BASE_DIR / "Deep_Features" / "resnet18_speech_features.csv"
EGG_CNN_CSV     = BASE_DIR / "Deep_Features" / "resnet18_egg_features.csv"

OUTPUT_FUSED_CSV = BASE_DIR / "fused_dataset_4class.csv"  # Full data

RANDOM_STATE = 42
# =========================================

def main():
    print("🔹 Loading feature files...")

    df_hand = pd.read_csv(HANDCRAFTED_CSV)
    df_speech = pd.read_csv(SPEECH_CNN_CSV)
    df_egg = pd.read_csv(EGG_CNN_CSV)

    # Rename CNN columns
    df_speech.rename(
        columns={c: f"speech_{c}" for c in df_speech.columns if c not in ["patient_id", "label"]},
        inplace=True
    )
    df_egg.rename(
        columns={c: f"egg_{c}" for c in df_egg.columns if c not in ["patient_id", "label"]},
        inplace=True
    )

    # Feature fusion
    print("\n🔹 Fusing features...")

    df_fused = pd.merge(df_hand, df_speech, on=["patient_id", "label"], how="inner")
    df_fused = pd.merge(df_fused, df_egg, on=["patient_id", "label"], how="inner")

    if "disease" in df_fused.columns:
        df_fused.drop(columns=["disease"], inplace=True)

    # 🔥 STEP 1: GROUP LABELS (1 & 3 → 1)
    print("\n🔹 Grouping labels (1 & 3 → structural/inflammatory)...")

    label_map = {
        0: 0,   # Healthy
        1: 1,   # Structural / Inflammatory
        3: 1,   # Structural / Inflammatory
        2: 2,   # Hyperfunctional
        4: 3    # Neurological
    }

    df_fused["label"] = df_fused["label"].map(label_map)

    df_fused.fillna(0, inplace=True)

    print("\n📊 Final class distribution (full data):")
    print(df_fused["label"].value_counts())

    df_fused.to_csv(OUTPUT_FUSED_CSV, index=False)

    print(f"\n✅ Fused dataset saved: {OUTPUT_FUSED_CSV}")
    print(f"   Shape: {df_fused.shape}")

if __name__ == "__main__":
    main()