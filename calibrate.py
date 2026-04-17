"""
Post-hoc logit calibration via prior correction.

Idea: the model predicts "neutral" for ~5500/7000 test samples because
Bengali speech features land in the neutral region of the classifier.
We estimate the per-class bias from the validation set and subtract it
from the test logits before argmax.

Correction: logit_adj[c] = logit[c] - log(p_model(c) / p_true(c))
where p_model(c) = fraction of val samples predicted as c
      p_true(c)  = true val prior (uniform = 1/7)
"""

import os, glob, numpy as np, torch, librosa, pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import warnings; warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────
DATA_ROOT    = "/home/aluno/Documentos/PAR/data"
MODEL_PATH   = "/home/aluno/Documentos/PAR/best_model"
SAMPLE_RATE  = 16000
MAX_DURATION = 6
BATCH_SIZE   = 16
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

ID2LABEL = {0:"angry",1:"disgust",2:"fear",3:"happy",4:"neutral",5:"sad",6:"surprise"}
LABEL2ID = {v:k for k,v in ID2LABEL.items()}
NUM_CLASSES = 7

CAFE_FOLDER_MAP = {
    "colкre":0,"colere":0,"dвgoцt":1,"degoat":1,"degout":1,
    "peur":2,"joie":3,"neutre":4,"tristesse":5,"surprise":6,
}
RAVDESS_MAP = {"01":4,"03":3,"04":5,"05":0,"06":2,"07":1,"08":6}  # "02" calm removed

# ── Data ─────────────────────────────────────
def load_audio(path):
    try:
        a, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=MAX_DURATION)
        return np.clip(a, -1.0, 1.0).astype(np.float32) if len(a) >= 1600 else None
    except: return None

def get_cafe(root):
    s = []
    for d in os.listdir(os.path.join(root,"anad_cafe")):
        ep = os.path.join(root,"anad_cafe",d)
        if not os.path.isdir(ep): continue
        lbl = CAFE_FOLDER_MAP.get(d.lower())
        if lbl is None: continue
        for f in glob.glob(os.path.join(ep,"**","*.wav"),recursive=True): s.append((f,lbl))
    return s

def get_asvp(root):
    s = []
    for sr in [os.path.join(root,"asvp_esd","ASVP_UPDATE","Audio"),
               os.path.join(root,"asvp_esd","ASVP_UPDATE","Bonus")]:
        for f in glob.glob(os.path.join(sr,"**","*.wav"),recursive=True):
            p = Path(f).stem.split("-")
            if len(p)>=3:
                lbl = RAVDESS_MAP.get(p[2])
                if lbl is not None: s.append((f,lbl))
    return s

def get_test(root):
    s = []
    for f in glob.glob(os.path.join(root,"test_set","SUBESCO","*.wav")):
        name = Path(f).stem.upper()
        for e,i in LABEL2ID.items():
            if f"_{e.upper()}_" in name or name.endswith(f"_{e.upper()}"):
                s.append((f,i)); break
    return s

all_train = get_cafe(DATA_ROOT) + get_asvp(DATA_ROOT)
_, val_data = train_test_split(all_train, test_size=0.1, random_state=42,
                               stratify=[l for _,l in all_train])
test_data = get_test(DATA_ROOT)
print(f"Val: {len(val_data)} | Test: {len(test_data)}")

# ── Dataset ───────────────────────────────────
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)

class AudioDS(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.ml = SAMPLE_RATE * MAX_DURATION
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        a = load_audio(path)
        if a is None: a = np.zeros(SAMPLE_RATE, dtype=np.float32)
        if len(a) > self.ml: a = a[:self.ml]
        inp = feature_extractor(a, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                                padding="max_length", max_length=self.ml,
                                truncation=True, return_attention_mask=True)
        return {"input_values": inp["input_values"].squeeze(0),
                "attention_mask": inp["attention_mask"].squeeze(0),
                "labels": torch.tensor(label, dtype=torch.long)}

val_loader  = DataLoader(AudioDS(val_data),  batch_size=BATCH_SIZE, num_workers=4)
test_loader = DataLoader(AudioDS(test_data), batch_size=BATCH_SIZE, num_workers=4)

# ── Load model ────────────────────────────────
print("Loading model...")
model = AutoModelForAudioClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

def get_logits(loader):
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            iv  = batch["input_values"].to(DEVICE)
            am  = batch["attention_mask"].to(DEVICE)
            lbl = batch["labels"]
            logits = model(input_values=iv, attention_mask=am).logits
            all_logits.append(logits.cpu())
            all_labels.extend(lbl.numpy())
    return torch.cat(all_logits), np.array(all_labels)

# ── Collect logits ────────────────────────────
print("Computing val logits for calibration...")
val_logits, val_labels = get_logits(val_loader)
print("Computing test logits...")
test_logits, test_labels = get_logits(test_loader)

# Uncalibrated results
raw_preds = test_logits.argmax(-1).numpy()
print(f"\nUncalibrated Test Accuracy: {accuracy_score(test_labels, raw_preds):.4f}")

# ── Prior correction ──────────────────────────
val_preds = val_logits.argmax(-1).numpy()
val_pred_counts = np.bincount(val_preds, minlength=NUM_CLASSES).astype(float)
val_true_counts = np.bincount(val_labels, minlength=NUM_CLASSES).astype(float)

# Bias = log(predicted rate / true rate) for each class
bias = np.log((val_pred_counts / val_pred_counts.sum()) /
              (val_true_counts / val_true_counts.sum()) + 1e-9)

print(f"\nEstimated class bias (val): {np.round(bias, 3)}")
print(f"  (positive = over-predicted, negative = under-predicted)")

# Apply different correction strengths and pick best on val
best_val_acc, best_alpha = 0, 0
for alpha in np.arange(0.0, 2.1, 0.1):
    adj = val_logits - alpha * torch.tensor(bias, dtype=torch.float32).unsqueeze(0)
    acc = accuracy_score(val_labels, adj.argmax(-1).numpy())
    if acc > best_val_acc:
        best_val_acc, best_alpha = acc, alpha

print(f"\nBest correction alpha={best_alpha:.1f} → Val Acc={best_val_acc:.4f}")

# Apply to test
bias_tensor = torch.tensor(bias, dtype=torch.float32).unsqueeze(0)
test_adj = test_logits - best_alpha * bias_tensor
cal_preds = test_adj.argmax(-1).numpy()

print(f"\nCalibrated Test Accuracy: {accuracy_score(test_labels, cal_preds):.4f}")
print("\nClassification Report:")
print(classification_report(test_labels, cal_preds, target_names=list(ID2LABEL.values())))
print(f"\nPred distribution: {Counter(ID2LABEL[p] for p in cal_preds)}")

# Save
pd.DataFrame({
    "file": [s[0] for s in test_data],
    "true_label": [ID2LABEL[l] for l in test_labels],
    "pred_label": [ID2LABEL[p] for p in cal_preds],
}).to_csv("/home/aluno/Documentos/PAR/test_results.csv", index=False)
print("\nResults saved to test_results.csv")
