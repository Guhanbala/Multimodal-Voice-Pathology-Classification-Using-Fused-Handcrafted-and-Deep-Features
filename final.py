import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# ================= CONFIGURATION =================
DATASET_FILENAME = "fused_dataset_4class.csv"
BASE_DIR = Path(__file__).parent.resolve()
DATASET_PATH = BASE_DIR / DATASET_FILENAME

RANDOM_STATE = 42
EPOCHS = 100
LR = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 20
DEVICE = "cpu"

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ================= MODEL DEFINITION =================
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, out_dim)
        )

    def forward(self, x):
        return self.net(x)

# ================= TRAIN FUNCTION =================
def train_model(model, X_tr, y_tr, X_val, y_val, stage_name):
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    X_tr = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
    y_tr = torch.tensor(y_tr, dtype=torch.long).to(DEVICE)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val = torch.tensor(y_val, dtype=torch.long).to(DEVICE)
    model = model.to(DEVICE)

    best_f1, patience_counter = 0, 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(X_tr)
        loss = criterion(logits, y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(X_val), dim=1).cpu().numpy()
            f1 = f1_score(y_val.cpu().numpy(), preds, average="macro", zero_division=0)

        if (epoch + 1) % 10 == 0:
            print(f"{stage_name} | Epoch {epoch+1:03d} | Loss {loss.item():.3f} | Val F1 {f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# ================= MAIN PIPELINE =================
def main():
    print(f"\n🔹 Loading dataset: {DATASET_PATH}")
    if not DATASET_PATH.exists():
        print(f"❌ Error: File not found at {DATASET_PATH}")
        return

    df = pd.read_csv(DATASET_PATH)

    # 1. AUTO-GENERATE SPLIT
    if 'split' not in df.columns:
        print("⚠️ 'split' column missing. Generating patient-level 80/20 split now...")
        unique_patients = df[['patient_id', 'label']].drop_duplicates()
        train_ids, test_ids = train_test_split(
            unique_patients['patient_id'], 
            test_size=0.2, 
            stratify=unique_patients['label'], 
            random_state=RANDOM_STATE
        )
        df['split'] = 'train'
        df.loc[df['patient_id'].isin(test_ids), 'split'] = 'test'
        print("✅ Split generated.")

    # 2. PREPARE DATA
    train_mask = df['split'] == 'train'
    test_mask = df['split'] == 'test'
    drop_cols = ["patient_id", "label", "split", "disease"]
    cols_to_drop = [c for c in drop_cols if c in df.columns]

    X_train = df[train_mask].drop(columns=cols_to_drop).values
    y_train = df[train_mask]["label"].values
    X_test = df[test_mask].drop(columns=cols_to_drop).values
    y_test = df[test_mask]["label"].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 3. TRAIN STAGE 1 (Binary)
    print("\n🚦 Training Stage 1...")
    y_tr_s1 = np.where(y_train == 0, 0, 1)
    X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(
        X_train, y_tr_s1, test_size=0.2, stratify=y_tr_s1, random_state=RANDOM_STATE
    )
    model_s1 = train_model(MLP(X_train.shape[1], 2), X_tr_sub, y_tr_sub, X_val_sub, y_val_sub, "Stage 1")

    # 4. TRAIN STAGE 2 (Pathology Types)
    print("\n🏥 Training Stage 2...")
    mask_p = y_train != 0
    X_p_train = X_train[mask_p]
    y_p_train = y_train[mask_p]
    y_s2_train = np.zeros_like(y_p_train)
    y_s2_train[y_p_train == 1] = 0
    y_s2_train[y_p_train == 2] = 1
    y_s2_train[y_p_train == 3] = 2

    X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(
        X_p_train, y_s2_train, test_size=0.2, stratify=y_s2_train, random_state=RANDOM_STATE
    )
    model_s2 = train_model(MLP(X_train.shape[1], 3), X_tr_sub, y_tr_sub, X_val_sub, y_val_sub, "Stage 2")

    # ================= EVALUATION =================
    print("\n🔮 FINAL EVALUATION (NO REJECTION)")
    model_s1.eval()
    model_s2.eval()

    # --- Hierarchy Level 1: Detection ---
    y_test_binary = np.where(y_test == 0, 0, 1)
    with torch.no_grad():
        logits1 = model_s1(torch.tensor(X_test, dtype=torch.float32).to(DEVICE))
        preds1 = torch.argmax(logits1, dim=1).cpu().numpy()
    
    acc1 = accuracy_score(y_test_binary, preds1)
    print(f"\n✅ Level 1 (Detection) Accuracy: {acc1:.4f}")
    
    cm1 = confusion_matrix(y_test_binary, preds1)
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=["Healthy", "Sick"])
    plt.figure()
    disp1.plot(cmap=plt.cm.Blues)
    plt.title(f"Level 1: Detection (Acc: {acc1:.2%})")
    plt.savefig("cm_level1_detection.png")

    # --- Hierarchy Level 2: Diagnosis (Only Sick) ---
    mask_test_p = y_test != 0
    if np.sum(mask_test_p) > 0:
        X_test_p = X_test[mask_test_p]
        y_test_p = y_test[mask_test_p]
        y_test_p_mapped = np.zeros_like(y_test_p)
        y_test_p_mapped[y_test_p == 1] = 0
        y_test_p_mapped[y_test_p == 2] = 1
        y_test_p_mapped[y_test_p == 3] = 2

        with torch.no_grad():
            logits2 = model_s2(torch.tensor(X_test_p, dtype=torch.float32).to(DEVICE))
            preds2 = torch.argmax(logits2, dim=1).cpu().numpy()

        acc2 = accuracy_score(y_test_p_mapped, preds2)
        print(f"✅ Level 2 (Diagnosis) Accuracy: {acc2:.4f}")
        
        cm2 = confusion_matrix(y_test_p_mapped, preds2)
        disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=["Struct", "Hyper", "Neuro"])
        plt.figure()
        disp2.plot(cmap=plt.cm.Blues)
        plt.title(f"Level 2: Diagnosis (Acc: {acc2:.2%})")
        plt.savefig("cm_level2_diagnosis.png")

    # --- Overall System ---
    final_preds = []
    with torch.no_grad():
        for i in range(len(X_test)):
            xi = torch.tensor(X_test[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            p1 = torch.argmax(model_s1(xi), dim=1).item()
            if p1 == 0:
                final_preds.append(0)
            else:
                p2 = torch.argmax(model_s2(xi), dim=1).item()
                final_preds.append(p2 + 1)

    acc_final = accuracy_score(y_test, final_preds)
    print(f"✅ Overall System Accuracy: {acc_final:.4f}")
    
    cm_final = confusion_matrix(y_test, final_preds)
    disp_final = ConfusionMatrixDisplay(confusion_matrix=cm_final, display_labels=["Healthy", "Struct", "Hyper", "Neuro"])
    plt.figure()
    disp_final.plot(cmap=plt.cm.Blues)
    plt.title(f"Overall System (Acc: {acc_final:.2%})")
    plt.savefig("cm_overall.png")
    
    print("\n📊 Saved confusion matrices: cm_level1_detection.png, cm_level2_diagnosis.png, cm_overall.png")

if __name__ == "__main__":
    main()