import os
import glob
import numpy as np
import torch
import librosa
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from collections import Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DATA_ROOT   = "/home/aluno/Documentos/PAR/data"
MODEL_NAME  = "microsoft/wavlm-large"
SAMPLE_RATE = 16000
MAX_DURATION = 6
BATCH_SIZE  = 8
EPOCHS      = 10
FREEZE_LAYERS = 16   # freeze first 16 of 24, fine-tune last 8 + classifier
LR_TRANSFORMER = 5e-6
LR_HEAD        = 5e-4
LABEL_SMOOTHING = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ID2LABEL = {0:"angry", 1:"disgust", 2:"fear", 3:"happy", 4:"neutral", 5:"sad", 6:"surprise"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
NUM_CLASSES = 7

CAFE_FOLDER_MAP = {
    "colкre": 0, "colere": 0,
    "dвgoцt": 1, "degoat": 1, "degout": 1,
    "peur": 2, "joie": 3, "neutre": 4, "tristesse": 5, "surprise": 6,
}
RAVDESS_MAP = {
    "01": 4,           # neutral
    # "02" calm → removed (acoustically similar to neutral but different, causes confusion)
    "03": 3,           # happy
    "04": 5,           # sad
    "05": 0,           # angry
    "06": 2,           # fearful
    "07": 1,           # disgust
    "08": 6,           # surprised
}

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ──────────────────────────────────────────────
# Audio
# ──────────────────────────────────────────────

def load_audio(path, augment=False):
    try:
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=MAX_DURATION)
        if len(audio) < 1600:
            return None
        if augment:
            # Gaussian noise
            if np.random.rand() < 0.4:
                audio += np.random.randn(len(audio)).astype(np.float32) * 0.005
            # Random gain
            if np.random.rand() < 0.4:
                audio *= np.random.uniform(0.8, 1.2)
            # Pitch shift (±2 semitones) — key for cross-lingual robustness
            if np.random.rand() < 0.3:
                steps = np.random.uniform(-2, 2)
                audio = librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=steps)
        return np.clip(audio, -1.0, 1.0).astype(np.float32)
    except Exception:
        return None

# ──────────────────────────────────────────────
# Dataset builders
# ──────────────────────────────────────────────

def get_cafe_samples(root):
    samples = []
    for d in os.listdir(os.path.join(root, "anad_cafe")):
        epath = os.path.join(root, "anad_cafe", d)
        if not os.path.isdir(epath):
            continue
        label = CAFE_FOLDER_MAP.get(d.lower())
        if label is None:
            continue
        for f in glob.glob(os.path.join(epath, "**", "*.wav"), recursive=True):
            samples.append((f, label))
    return samples

def get_asvp_samples(root):
    samples = []
    for search_root in [
        os.path.join(root, "asvp_esd", "ASVP_UPDATE", "Audio"),
        os.path.join(root, "asvp_esd", "ASVP_UPDATE", "Bonus"),
    ]:
        for f in glob.glob(os.path.join(search_root, "**", "*.wav"), recursive=True):
            parts = Path(f).stem.split("-")
            if len(parts) >= 3:
                label = RAVDESS_MAP.get(parts[2])
                if label is not None:
                    samples.append((f, label))
    return samples

def get_test_samples(root):
    samples = []
    for f in glob.glob(os.path.join(root, "test_set", "SUBESCO", "*.wav")):
        name = Path(f).stem.upper()
        for emo_str, emo_id in LABEL2ID.items():
            if f"_{emo_str.upper()}_" in name or name.endswith(f"_{emo_str.upper()}"):
                samples.append((f, emo_id))
                break
    return samples

# ──────────────────────────────────────────────
# Load & split
# ──────────────────────────────────────────────
print("\nBuilding dataset...")
cafe_samples = get_cafe_samples(DATA_ROOT)
asvp_samples = get_asvp_samples(DATA_ROOT)
test_samples = get_test_samples(DATA_ROOT)
print(f"CaFE: {len(cafe_samples)} | ASVP: {len(asvp_samples)} | Test: {len(test_samples)}")

all_train = cafe_samples + asvp_samples
train_data, val_data = train_test_split(
    all_train, test_size=0.1, random_state=42,
    stratify=[l for _, l in all_train]
)
print(f"Train: {len(train_data)} | Val: {len(val_data)}")

# WeightedRandomSampler for class balance
train_labels = [l for _, l in train_data]
class_counts  = Counter(train_labels)
total = sum(class_counts.values())
sample_weights = [total / class_counts[l] for _, l in train_data]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class SpeechEmotionDataset(Dataset):
    def __init__(self, samples, feature_extractor, augment=False):
        self.samples = samples
        self.feature_extractor = feature_extractor
        self.augment = augment
        self.max_length = SAMPLE_RATE * MAX_DURATION

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        audio = load_audio(path, augment=self.augment)
        if audio is None:
            audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        if len(audio) > self.max_length:
            start = np.random.randint(0, len(audio) - self.max_length) if self.augment else 0
            audio = audio[start:start + self.max_length]

        inputs = self.feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
        )
        return {
            "input_values":   inputs["input_values"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# ──────────────────────────────────────────────
# Model + freezing (generic for WavLM/wav2vec2)
# ──────────────────────────────────────────────
print(f"\nLoading model: {MODEL_NAME}")
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForAudioClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    label2id=LABEL2ID,
    id2label=ID2LABEL,
    ignore_mismatched_sizes=True,
).to(DEVICE)

# Freeze CNN feature extractor (first module before transformer)
def get_encoder_layers(model):
    for attr in ["wavlm", "wav2vec2", "hubert"]:
        base = getattr(model, attr, None)
        if base is not None:
            for p in base.feature_extractor.parameters():
                p.requires_grad = False
            return base.encoder.layers
    raise ValueError("Unknown model architecture")

encoder_layers = get_encoder_layers(model)
for i, layer in enumerate(encoder_layers):
    if i < FREEZE_LAYERS:
        for p in layer.parameters():
            p.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"Trainable: {trainable:,} | Frozen: {frozen:,}")

transformer_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and "classifier" not in n and "projector" not in n]
head_params = [p for n, p in model.named_parameters()
               if p.requires_grad and ("classifier" in n or "projector" in n)]

optimizer = torch.optim.AdamW([
    {"params": transformer_params, "lr": LR_TRANSFORMER},
    {"params": head_params,        "lr": LR_HEAD},
], weight_decay=0.01)

# ──────────────────────────────────────────────
# DataLoaders
# ──────────────────────────────────────────────
train_dataset = SpeechEmotionDataset(train_data, feature_extractor, augment=True)
val_dataset   = SpeechEmotionDataset(val_data,   feature_extractor, augment=False)
test_dataset  = SpeechEmotionDataset(test_samples, feature_extractor, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 5,
    num_training_steps=total_steps,
)
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

# ──────────────────────────────────────────────
# Train / eval
# ──────────────────────────────────────────────

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            iv  = batch["input_values"].to(DEVICE)
            am  = batch["attention_mask"].to(DEVICE)
            lbl = batch["labels"].to(DEVICE)
            logits = model(input_values=iv, attention_mask=am).logits
            total_loss += loss_fn(logits, lbl).item()
            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())
    return total_loss / len(loader), accuracy_score(all_labels, all_preds), all_preds, all_labels


print("\nStarting training...")
best_val_acc = 0
scaler = torch.amp.GradScaler("cuda")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in pbar:
        iv  = batch["input_values"].to(DEVICE)
        am  = batch["attention_mask"].to(DEVICE)
        lbl = batch["labels"].to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(input_values=iv, attention_mask=am).logits
            loss = loss_fn(logits, lbl)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(train_loader)
    val_loss, val_acc, _, _ = evaluate(model, val_loader)
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained("/home/aluno/Documentos/PAR/best_model")
        feature_extractor.save_pretrained("/home/aluno/Documentos/PAR/best_model")
        print(f"  → Saved best model (val_acc={val_acc:.4f})")

# ──────────────────────────────────────────────
# Test evaluation
# ──────────────────────────────────────────────
print("\nLoading best model for test evaluation...")
best_model = AutoModelForAudioClassification.from_pretrained(
    "/home/aluno/Documentos/PAR/best_model"
).to(DEVICE)

_, test_acc, test_preds, test_labels_out = evaluate(best_model, test_loader)
print(f"\nTest Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(test_labels_out, test_preds, target_names=list(ID2LABEL.values())))

pd.DataFrame({
    "file":       [s[0] for s in test_samples],
    "true_label": [ID2LABEL[l] for l in test_labels_out],
    "pred_label": [ID2LABEL[p] for p in test_preds],
}).to_csv("/home/aluno/Documentos/PAR/test_results.csv", index=False)
print("Results saved to test_results.csv")
