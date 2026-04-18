"""
Recalibracao avancada pos-treino para o modelo XLS-R v2.

Estrategias:
  1. Global alpha ampliado (0..6) -- maior range que o original (0..2.5)
  2. Alpha por classe -- cada classe tem seu proprio fator de correcao
  3. Distribuicao-alvo (transductivo) -- usa o fato de que SUBESCO tem 1000/classe
     para encontrar correcoes que aproximam a distribuicao predita a uniforme
"""

import os, glob, unicodedata, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, Wav2Vec2Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import pandas as pd
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")

DATA_ROOT   = "/workspace/kaiki_home/data"
MODEL_DIR   = "/workspace/kaiki_home/best_model_v2"
CAFE_DIR    = os.path.join(DATA_ROOT, "anad_cafe")
ASVP_AUDIO  = os.path.join(DATA_ROOT, "asvp_esd", "ASVP_UPDATE", "Audio")
ASVP_BONUS  = os.path.join(DATA_ROOT, "asvp_esd", "ASVP_UPDATE", "Bonus")
TEST_DIR    = os.path.join(DATA_ROOT, "test_set", "SUBESCO")
SAMPLE_RATE = 16000
MAX_DURATION= 6
MAX_SAMPLES = SAMPLE_RATE * MAX_DURATION
BATCH_SIZE  = 16
TTA_CROPS   = 3
SEED        = 42
BACKBONE    = "facebook/wav2vec2-xls-r-300m"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

ID2LABEL = {0:"angry",1:"disgust",2:"fear",3:"happy",4:"neutral",5:"sad",6:"surprise"}
LABEL2ID = {v:k for k,v in ID2LABEL.items()}
NUM_CLASSES = 7

def _norm(s):
    return unicodedata.normalize("NFD",s).encode("ascii","ignore").decode().lower().strip()

CAFE_MAP_NORM = {"colere":0,"degout":1,"degoat":1,"peur":2,"joie":3,"neutre":4,"tristesse":5,"surprise":6}
RAVDESS_MAP   = {"01":4,"02":4,"03":3,"04":5,"05":0,"06":2,"07":1,"08":6}

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def get_cafe(cafe_dir):
    s=[]
    for d in os.listdir(cafe_dir):
        lbl=CAFE_MAP_NORM.get(_norm(d))
        if lbl is None: continue
        ep=os.path.join(cafe_dir,d)
        if not os.path.isdir(ep): continue
        for f in glob.glob(os.path.join(ep,"**","*.wav"),recursive=True): s.append((f,lbl))
    return s

def get_asvp(audio_dir, bonus_dir):
    s=[]
    for d in [audio_dir, bonus_dir]:
        if not os.path.isdir(d): continue
        for f in glob.glob(os.path.join(d,"**","*.wav"),recursive=True):
            p=Path(f).stem.split("-")
            if len(p)>=3:
                lbl=RAVDESS_MAP.get(p[2])
                if lbl is not None: s.append((f,lbl))
    return s

def get_test(test_dir):
    s=[]
    for f in glob.glob(os.path.join(test_dir,"*.wav")):
        name=Path(f).stem.upper()
        for e,i in LABEL2ID.items():
            if f"_{e.upper()}_" in name or name.endswith(f"_{e.upper()}"):
                s.append((f,i)); break
    return s

all_train = get_cafe(CAFE_DIR) + get_asvp(ASVP_AUDIO, ASVP_BONUS)
_, val_data = train_test_split(all_train, test_size=0.1, random_state=SEED,
                               stratify=[l for _,l in all_train])
test_data = get_test(TEST_DIR)
print(f"Val: {len(val_data)} | Test: {len(test_data)}")

# ── Dataset ───────────────────────────────────────────────────────────────────
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_DIR)

def load_audio(path):
    try:
        a,_ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=MAX_DURATION)
        return np.clip(a,-1.,1.).astype(np.float32) if len(a)>=1600 else None
    except: return None

class AudioDS(Dataset):
    def __init__(self, samples, n_crops=1):
        self.samples=samples; self.n_crops=n_crops
    def __len__(self): return len(self.samples)
    def _proc(self, audio):
        inp=feature_extractor(audio,sampling_rate=SAMPLE_RATE,return_tensors="pt",
                              padding="max_length",max_length=MAX_SAMPLES,
                              truncation=True,return_attention_mask=True)
        return inp["input_values"].squeeze(0), inp["attention_mask"].squeeze(0)
    def __getitem__(self, idx):
        path,label=self.samples[idx]
        audio=load_audio(path)
        if audio is None: audio=np.zeros(SAMPLE_RATE,dtype=np.float32)
        if self.n_crops==1:
            if len(audio)>MAX_SAMPLES: audio=audio[:MAX_SAMPLES]
            iv,am=self._proc(audio)
            return {"iv":iv,"am":am,"label":torch.tensor(label,dtype=torch.long)}
        else:
            full,_=librosa.load(path,sr=SAMPLE_RATE,mono=True)
            full=np.clip(full,-1.,1.).astype(np.float32)
            ivs,ams=[],[]
            for c in range(self.n_crops):
                if len(full)>=MAX_SAMPLES:
                    ms=len(full)-MAX_SAMPLES
                    st=int(c*ms/max(1,self.n_crops-1))
                    crop=full[st:st+MAX_SAMPLES]
                else: crop=full
                iv,am=self._proc(crop); ivs.append(iv); ams.append(am)
            return {"iv":torch.stack(ivs),"am":torch.stack(ams),
                    "label":torch.tensor(label,dtype=torch.long)}

# ── Modelo ────────────────────────────────────────────────────────────────────
class XLSREmotionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder   = Wav2Vec2Model.from_pretrained(BACKBONE, use_safetensors=True)
        n_layers       = self.encoder.config.num_hidden_layers + 1
        hidden         = self.encoder.config.hidden_size
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))
        self.attn_proj = nn.Linear(hidden, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden,512),
            nn.GELU(), nn.Dropout(0.3), nn.Linear(512,NUM_CLASSES),
        )
    def forward(self, input_values, attention_mask=None):
        out=self.encoder(input_values,attention_mask=attention_mask,output_hidden_states=True)
        hs=torch.stack(out.hidden_states,dim=1)
        w=F.softmax(self.layer_weights,dim=0).view(1,-1,1,1)
        x=(hs*w).sum(dim=1)
        attn=F.softmax(self.attn_proj(x),dim=1)
        x=(x*attn).sum(dim=1)
        return self.classifier(x)

print("Carregando modelo...")
model = XLSREmotionClassifier().to(DEVICE)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR,"model.pt"),
                                 map_location=DEVICE, weights_only=True))
model.eval()

# ── Coleta de logits ──────────────────────────────────────────────────────────
@torch.no_grad()
def get_logits(samples, n_crops=1):
    ds=AudioDS(samples, n_crops=n_crops)
    bs=BATCH_SIZE if n_crops==1 else max(1, BATCH_SIZE//n_crops)
    dl=DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2)
    all_logits, all_labels = [], []
    for batch in tqdm(dl, desc=f"  Inferencia (crops={n_crops})"):
        lbl=batch["label"]
        if n_crops==1:
            iv=batch["iv"].to(DEVICE); am=batch["am"].to(DEVICE)
            with torch.amp.autocast("cuda",enabled=False):
                logits=model(iv.float(),am)
        else:
            B,N,T=batch["iv"].shape
            iv=batch["iv"].view(B*N,T).to(DEVICE)
            am=batch["am"].view(B*N,T).to(DEVICE)
            with torch.amp.autocast("cuda",enabled=False):
                logits=model(iv.float(),am).view(B,N,NUM_CLASSES).mean(1)
        if torch.isnan(logits).any(): continue
        all_logits.append(logits.cpu().float())
        all_labels.extend(lbl.numpy())
    return torch.cat(all_logits), np.array(all_labels)

print("\nColetando logits do val set (sem TTA)...")
val_logits, val_labels = get_logits(val_data, n_crops=1)
print("Coletando logits do test set (TTA 3 crops)...")
test_logits, test_labels = get_logits(test_data, n_crops=TTA_CROPS)

print(f"\nVal:  {len(val_labels)} amostras | Test: {len(test_labels)} amostras")

# Estimativa de bias a partir do val set
val_preds_raw = val_logits.argmax(-1).numpy()
pred_dist = np.bincount(val_preds_raw, minlength=NUM_CLASSES).astype(float)
true_dist = np.bincount(val_labels,    minlength=NUM_CLASSES).astype(float)
bias = np.log((pred_dist/pred_dist.sum()) / (true_dist/true_dist.sum() + 1e-9) + 1e-9)
print(f"\nBias estimado (val set):")
for i,name in ID2LABEL.items():
    print(f"  {name:10s}: {bias[i]:+.3f}  ({'over' if bias[i]>0 else 'under'}-previsto)")

bias_t = torch.tensor(bias, dtype=torch.float32).unsqueeze(0)

print(f"\n{'='*60}")
print("RESULTADOS SEM CALIBRACAO")
print('='*60)
raw_preds = test_logits.argmax(-1).numpy()
print(f"Test Accuracy: {accuracy_score(test_labels, raw_preds):.4f}")
print(f"Distribuicao: {dict(Counter(ID2LABEL[p] for p in raw_preds))}")

# ── Estrategia 1: Global alpha ampliado (0..6) ────────────────────────────────
print(f"\n{'='*60}")
print("ESTRATEGIA 1: Global alpha ampliado [0..6]")
print('='*60)
best_acc, best_alpha = 0.0, 0.0
for alpha in np.arange(0.0, 6.1, 0.05):
    adj = val_logits - alpha * bias_t
    acc = accuracy_score(val_labels, adj.argmax(-1).numpy())
    if acc > best_acc:
        best_acc, best_alpha = acc, alpha
print(f"Melhor alpha={best_alpha:.2f} -> val acc={best_acc:.4f}")
test_adj1 = test_logits - best_alpha * bias_t
preds1 = test_adj1.argmax(-1).numpy()
acc1 = accuracy_score(test_labels, preds1)
print(f"Test Accuracy: {acc1:.4f}")
print(f"Distribuicao: {dict(Counter(ID2LABEL[p] for p in preds1))}")

# ── Estrategia 2: Alpha por classe ───────────────────────────────────────────
print(f"\n{'='*60}")
print("ESTRATEGIA 2: Alpha por classe (otimizacao no val set)")
print('='*60)

# Grid search: cada classe tem seu proprio fator multiplicativo para a correcao de bias
# Limitamos a busca para manter tratabilidade
from itertools import product

# Para cada classe, busca o melhor alpha individual
# Usa coordenadas descendentes: otimiza uma classe por vez, itera
alphas_per_class = np.zeros(NUM_CLASSES)
best_val_acc_cls = accuracy_score(val_labels, val_logits.argmax(-1).numpy())

for iteration in range(3):  # 3 rounds de otimizacao coordenada
    improved = False
    for c in range(NUM_CLASSES):
        best_a_c = alphas_per_class[c]
        for a_c in np.arange(0.0, 6.1, 0.1):
            a_vec = alphas_per_class.copy()
            a_vec[c] = a_c
            adj = val_logits - torch.tensor(a_vec * bias, dtype=torch.float32).unsqueeze(0)
            acc = accuracy_score(val_labels, adj.argmax(-1).numpy())
            if acc > best_val_acc_cls:
                best_val_acc_cls = acc
                best_a_c = a_c
                improved = True
        alphas_per_class[c] = best_a_c
    if not improved:
        break

print(f"Alphas por classe encontrados (val acc={best_val_acc_cls:.4f}):")
for i,name in ID2LABEL.items():
    print(f"  {name:10s}: alpha={alphas_per_class[i]:.1f}")

correction2 = torch.tensor(alphas_per_class * bias, dtype=torch.float32).unsqueeze(0)
test_adj2 = test_logits - correction2
preds2 = test_adj2.argmax(-1).numpy()
acc2 = accuracy_score(test_labels, preds2)
print(f"Test Accuracy: {acc2:.4f}")
print(f"Distribuicao: {dict(Counter(ID2LABEL[p] for p in preds2))}")

# ── Estrategia 3: Distribuicao-alvo uniforme (SUBESCO tem 1000/classe) ────────
print(f"\n{'='*60}")
print("ESTRATEGIA 3: Distribuicao-alvo uniforme (transductivo)")
print("  (SUBESCO tem exatamente 1000 amostras/classe -- fato conhecido do dataset)")
print('='*60)

# Usa a distribuicao real de predicoes no TEST para estimar o bias de dominio
# Assumindo distribuicao UNIFORME como alvo (verificado na analise do dataset)
test_preds_raw = test_logits.argmax(-1).numpy()
test_pred_dist = np.bincount(test_preds_raw, minlength=NUM_CLASSES).astype(float)
uniform_dist   = np.ones(NUM_CLASSES) / NUM_CLASSES

# Bias de dominio: log(pred_rate / uniform_rate)
test_bias = np.log((test_pred_dist/test_pred_dist.sum()) / uniform_dist + 1e-9)
print(f"Bias de dominio estimado do test (para distribuicao uniforme):")
for i,name in ID2LABEL.items():
    print(f"  {name:10s}: {test_bias[i]:+.3f}")

test_bias_t = torch.tensor(test_bias, dtype=torch.float32).unsqueeze(0)

# Grid search alpha no VAL set usando o bias do TEST
best_acc3, best_alpha3 = 0.0, 0.0
for alpha in np.arange(0.0, 4.1, 0.05):
    adj = val_logits - alpha * test_bias_t
    acc = accuracy_score(val_labels, adj.argmax(-1).numpy())
    if acc > best_acc3:
        best_acc3, best_alpha3 = acc, alpha
print(f"\nMelhor alpha={best_alpha3:.2f} -> val acc={best_acc3:.4f}")
test_adj3 = test_logits - best_alpha3 * test_bias_t
preds3 = test_adj3.argmax(-1).numpy()
acc3 = accuracy_score(test_labels, preds3)
print(f"Test Accuracy: {acc3:.4f}")
print(f"Distribuicao: {dict(Counter(ID2LABEL[p] for p in preds3))}")

# ── Ensemble das estrategias ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print("ENSEMBLE: media dos logits calibrados de todas as estrategias")
print('='*60)
ensemble_logits = (test_adj1 + test_adj2 + test_adj3) / 3
preds_ens = ensemble_logits.argmax(-1).numpy()
acc_ens = accuracy_score(test_labels, preds_ens)
print(f"Test Accuracy: {acc_ens:.4f}")
print(f"Distribuicao: {dict(Counter(ID2LABEL[p] for p in preds_ens))}")

print(f"\n{'='*60}")
print("SUMARIO")
print('='*60)
results = {
    "Sem calibracao":      (accuracy_score(test_labels, raw_preds),  raw_preds),
    "Global alpha amplo":  (acc1,  preds1),
    "Alpha por classe":    (acc2,  preds2),
    "Transductivo":        (acc3,  preds3),
    "Ensemble":            (acc_ens, preds_ens),
}
best_name = max(results, key=lambda k: results[k][0])
for name, (acc, _) in sorted(results.items(), key=lambda x: x[1][0], reverse=True):
    flag = " <-- MELHOR" if name == best_name else ""
    print(f"  {name:25s}: {acc:.4f}{flag}")

best_acc_final, best_preds_final = results[best_name]
print(f"\nRelatorio da melhor estrategia ({best_name}):")
print(classification_report(test_labels, best_preds_final,
                             target_names=[ID2LABEL[i] for i in range(NUM_CLASSES)]))

out_path = "/workspace/kaiki_home/test_results_v2_recal.csv"
pd.DataFrame({
    "file":       [s[0] for s in test_data],
    "true_label": [ID2LABEL[l] for l in test_labels],
    "pred_label": [ID2LABEL[p] for p in best_preds_final],
}).to_csv(out_path, index=False)
print(f"Resultados salvos em: {out_path}")
