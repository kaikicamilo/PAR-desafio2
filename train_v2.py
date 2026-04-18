import os
import glob
import random
import unicodedata
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoFeatureExtractor, Wav2Vec2Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from collections import Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

DATA_ROOT    = "/workspace/kaiki_home/data"
CAFE_DIR     = os.path.join(DATA_ROOT, "anad_cafe")
ASVP_AUDIO   = os.path.join(DATA_ROOT, "asvp_esd", "ASVP_UPDATE", "Audio")
ASVP_BONUS   = os.path.join(DATA_ROOT, "asvp_esd", "ASVP_UPDATE", "Bonus")
TEST_DIR     = os.path.join(DATA_ROOT, "test_set", "SUBESCO")
SAVE_DIR     = "/workspace/kaiki_home/best_model_v2"

BACKBONE     = "facebook/wav2vec2-xls-r-300m"  # 128 linguas, inclui Bengali
SAMPLE_RATE  = 16000
MAX_DURATION = 6                                # segundos
MAX_SAMPLES  = SAMPLE_RATE * MAX_DURATION
BATCH_SIZE   = 8
EPOCHS       = 12
LR_ENCODER   = 5e-6
LR_HEAD      = 5e-4
LABEL_SMOOTH = 0.1
TTA_CROPS    = 3
SEED         = 42
RESUME_EPOCH = 0

# Congelar progressivamente:
# epocas 1..FREEZE_ALL  -> encoder todo congelado
# epocas FREEZE_ALL+1..FREEZE_PARTIAL -> ultimas UNFREEZE_LAYERS camadas liberadas
# epocas FREEZE_PARTIAL+1.. -> tudo liberado
FREEZE_ALL     = 2
FREEZE_PARTIAL = 6
UNFREEZE_LAYERS = 8   # camadas que ficam liberadas na fase 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ID2LABEL = {0:"angry", 1:"disgust", 2:"fear", 3:"happy", 4:"neutral", 5:"sad", 6:"surprise"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
NUM_CLASSES = 7

def _norm(s: str) -> str:
    """Remove acentos e lowercase: 'Colère' -> 'colere'"""
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower().strip()

# Chaves ja normalizadas (sem acentos)
CAFE_MAP_NORM = {
    "colere":    0,   # colère  (raiva)
    "degout":    1,   # dégoût  (nojo)
    "degoat":    1,   # variante de encoding
    "peur":      2,   # medo
    "joie":      3,   # alegria
    "neutre":    4,   # neutro
    "tristesse": 5,   # tristeza
    "surprise":  6,
}

RAVDESS_MAP = {
    "01": 4,    # neutral
    "02": 4,    # calm -> neutral (mesmo arousal baixo, melhora cobertura de neutral)
    "03": 3,    # happy
    "04": 5,    # sad
    "05": 0,    # angry
    "06": 2,    # fearful
    "07": 1,    # disgust
    "08": 6,    # surprised
    # "09"-"12": emocoes extras do ASVP_ESD sem correspondencia no nosso schema -- excluidos
}


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

set_seed(SEED)
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")


# ── Carregamento de dados ─────────────────────────────────────────────────────
def get_cafe_samples(cafe_dir):
    samples = []
    if not os.path.isdir(cafe_dir):
        print(f"[AVISO] CaFE nao encontrado em: {cafe_dir}")
        return samples
    skipped = []
    for d in os.listdir(cafe_dir):
        label = CAFE_MAP_NORM.get(_norm(d))
        if label is None:
            skipped.append(d)
            continue
        epath = os.path.join(cafe_dir, d)
        if not os.path.isdir(epath):
            continue
        for f in glob.glob(os.path.join(epath, "**", "*.wav"), recursive=True):
            samples.append((f, label))
    if skipped:
        print(f"  [CaFE] Pastas ignoradas (sem label): {skipped}")
    print(f"CaFE: {len(samples)} amostras")
    return samples


def get_asvp_samples(audio_dir, bonus_dir):
    samples = []
    for search_dir in [audio_dir, bonus_dir]:
        if not os.path.isdir(search_dir):
            continue
        for f in glob.glob(os.path.join(search_dir, "**", "*.wav"), recursive=True):
            parts = Path(f).stem.split("-")
            if len(parts) >= 3:
                label = RAVDESS_MAP.get(parts[2])
                if label is not None:
                    samples.append((f, label))
    print(f"ASVP_ESD: {len(samples)} amostras")
    return samples


def get_test_samples(test_dir):
    samples = []
    if not os.path.isdir(test_dir):
        print(f"[AVISO] Test set nao encontrado em: {test_dir}")
        return samples
    for f in glob.glob(os.path.join(test_dir, "*.wav")):
        name = Path(f).stem.upper()
        for emo_str, emo_id in LABEL2ID.items():
            if f"_{emo_str.upper()}_" in name or name.endswith(f"_{emo_str.upper()}"):
                samples.append((f, emo_id))
                break
    print(f"SUBESCO test: {len(samples)} amostras")
    return samples


# ── Augmentation ──────────────────────────────────────────────────────────────
def augment_audio(audio: np.ndarray) -> np.ndarray:
    """Pipeline de augmentation focado em robustez cross-lingual."""
    # Ruido gaussiano
    if random.random() < 0.4:
        audio = audio + np.random.randn(len(audio)).astype(np.float32) * 0.005

    # Ganho aleatorio
    if random.random() < 0.4:
        audio = audio * np.random.uniform(0.8, 1.2)

    # Speed perturbation (0.9x ou 1.1x) -- altera ritmo sem mudar tonalidade
    # Critico para robustez de prosodia cross-lingual
    if random.random() < 0.4:
        rate = np.random.choice([0.9, 1.1])
        audio = librosa.effects.time_stretch(audio, rate=rate)
        # Re-fixar tamanho apos stretch
        if len(audio) > MAX_SAMPLES:
            audio = audio[:MAX_SAMPLES]
        elif len(audio) < MAX_SAMPLES:
            audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)))

    # Pitch shift (+-2 semitons) -- invariancia de tonalidade
    if random.random() < 0.3:
        steps = np.random.uniform(-2, 2)
        audio = librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=steps)

    # Time masking (SpecAugment simplificado no waveform)
    if random.random() < 0.3:
        mask_len = int(np.random.uniform(0.05, 0.2) * len(audio))
        start    = np.random.randint(0, max(1, len(audio) - mask_len))
        audio[start:start + mask_len] = 0.0

    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def load_audio(path, augment=False):
    try:
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=MAX_DURATION)
        if len(audio) < 1600:
            return None
        if augment:
            audio = augment_audio(audio)
        return np.clip(audio, -1.0, 1.0).astype(np.float32)
    except Exception:
        return None


# ── Dataset ───────────────────────────────────────────────────────────────────
class SpeechEmotionDataset(Dataset):
    def __init__(self, samples, feature_extractor, augment=False, n_crops=1):
        self.samples           = samples
        self.feature_extractor = feature_extractor
        self.augment           = augment
        self.n_crops           = n_crops  # TTA: retorna multiplos crops

    def __len__(self):
        return len(self.samples)

    def _process(self, audio):
        inp = self.feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_SAMPLES,
            truncation=True,
            return_attention_mask=True,
        )
        return (inp["input_values"].squeeze(0),
                inp["attention_mask"].squeeze(0))

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        audio = load_audio(path, augment=self.augment)
        if audio is None:
            audio = np.zeros(SAMPLE_RATE, dtype=np.float32)

        if self.n_crops == 1:
            # Treino: crop aleatorio se augment, senao inicio
            if self.augment and len(audio) >= MAX_SAMPLES:
                start = random.randint(0, len(audio) - MAX_SAMPLES)
                audio = audio[start:start + MAX_SAMPLES]
            iv, am = self._process(audio)
            return {"input_values": iv, "attention_mask": am,
                    "labels": torch.tensor(label, dtype=torch.long)}
        else:
            # TTA: n_crops recortes espacados uniformemente
            full_audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            full_audio = np.clip(full_audio, -1.0, 1.0).astype(np.float32)
            crops_iv, crops_am = [], []
            for c in range(self.n_crops):
                if len(full_audio) >= MAX_SAMPLES:
                    max_start = len(full_audio) - MAX_SAMPLES
                    start = int(c * max_start / max(1, self.n_crops - 1))
                    crop = full_audio[start:start + MAX_SAMPLES]
                else:
                    crop = full_audio
                iv, am = self._process(crop)
                crops_iv.append(iv)
                crops_am.append(am)
            return {"input_values": torch.stack(crops_iv),
                    "attention_mask": torch.stack(crops_am),
                    "labels": torch.tensor(label, dtype=torch.long)}


# ── Modelo: XLS-R + WLC + Attention Pooling ──────────────────────────────────
class XLSREmotionClassifier(nn.Module):
    """
    XLS-R 300M (128 linguas) como backbone com:
    - Weighted Layer Combination: aprende pesos para todas as 25 camadas
    - Attention Pooling: pondera frames pelo conteudo emocional
    - Classificador com LayerNorm + 2 camadas
    """
    def __init__(self, backbone_name=BACKBONE, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder   = Wav2Vec2Model.from_pretrained(backbone_name, use_safetensors=True)
        n_layers       = self.encoder.config.num_hidden_layers + 1  # +1 embedding
        hidden         = self.encoder.config.hidden_size             # 1024

        # Pesos aprendíveis para fusao de camadas (init=0 -> softmax uniforme)
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))

        # Attention pooling temporal
        self.attn_proj = nn.Linear(hidden, 1)

        # Cabeca classificadora
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    # ── Controle de congelamento ──────────────────────────────────────────────
    def freeze_all(self):
        """Congela CNN e todos os transformers."""
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_last_n(self, n):
        """Libera as ultimas n camadas transformer (CNN permanece congelada)."""
        # CNN sempre congelada
        for p in self.encoder.feature_extractor.parameters():
            p.requires_grad = False
        # Camadas transformer: libera as ultimas n
        layers = self.encoder.encoder.layers
        for i, layer in enumerate(layers):
            for p in layer.parameters():
                p.requires_grad = (i >= len(layers) - n)
        # Layer norm final sempre liberada
        for p in self.encoder.encoder.layer_norm.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        """Libera tudo menos CNN (CNN do feature extractor permanece congelada)."""
        for p in self.encoder.feature_extractor.parameters():
            p.requires_grad = False
        for p in self.encoder.encoder.parameters():
            p.requires_grad = True

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, input_values, attention_mask=None):
        out = self.encoder(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Weighted Layer Combination: [B, n_layers, T, H] -> [B, T, H]
        hidden_states = torch.stack(out.hidden_states, dim=1)
        w = F.softmax(self.layer_weights, dim=0).view(1, -1, 1, 1)
        x = (hidden_states * w).sum(dim=1)    # [B, T, H]

        # Attention Pooling simples: sem masked_fill (evita NaN por -inf)
        # Todos os audios sao padded para MAX_SAMPLES, entao nao ha frames invalidos
        attn = F.softmax(self.attn_proj(x), dim=1)   # [B, T, 1]
        x    = (x * attn).sum(dim=1)                  # [B, H]

        return self.classifier(x)             # [B, num_classes]


# ── Avaliacao ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, return_logits=False):
    """Roda em fp32 para evitar overflow de fp16 quando encoder esta descongelado."""
    model.eval()
    all_preds, all_labels, all_logits = [], [], []
    total_loss = 0.0
    n_batches  = 0
    loss_fn_eval = nn.CrossEntropyLoss()
    for batch in loader:
        iv  = batch["input_values"].to(DEVICE)
        am  = batch["attention_mask"].to(DEVICE)
        lbl = batch["labels"].to(DEVICE)
        # fp32 explicitamente -- evita NaN por overflow fp16
        with torch.amp.autocast("cuda", enabled=False):
            logits = model(iv.float(), am)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            continue  # pula batch corrompido
        batch_loss = loss_fn_eval(logits, lbl)
        if not (torch.isnan(batch_loss) or torch.isinf(batch_loss)):
            total_loss += batch_loss.item()
            n_batches  += 1
        all_preds.extend(logits.argmax(-1).cpu().numpy())
        all_labels.extend(lbl.cpu().numpy())
        if return_logits:
            all_logits.append(logits.cpu().float())
    acc  = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    loss = total_loss / max(1, n_batches)
    if return_logits:
        return loss, acc, all_preds, all_labels, torch.cat(all_logits)
    return loss, acc, all_preds, all_labels


# ── Calibracao de prior ────────────────────────────────────────────────────────
@torch.no_grad()
def calibrate_and_predict(model, val_loader, test_loader_tta):
    """
    1. Coleta logits no val set -> estima bias de classe
    2. Grid search alpha [0, 2] para maximizar val acc
    3. Aplica correcao no test com TTA (3 crops, media dos logits)
    """
    model.eval()

    # --- Logits do val set ---
    val_logits_list, val_labels_list = [], []
    for batch in tqdm(val_loader, desc="Logits val"):
        iv = batch["input_values"].to(DEVICE)
        am = batch["attention_mask"].to(DEVICE)
        with torch.amp.autocast("cuda", enabled=False):
            logits = model(iv.float(), am)
        val_logits_list.append(logits.cpu().float())
        val_labels_list.extend(batch["labels"].numpy())
    val_logits = torch.cat(val_logits_list)
    val_labels = np.array(val_labels_list)

    # --- Bias de classe ---
    val_preds       = val_logits.argmax(-1).numpy()
    pred_dist       = np.bincount(val_preds, minlength=NUM_CLASSES).astype(float)
    true_dist       = np.bincount(val_labels, minlength=NUM_CLASSES).astype(float)
    bias = np.log((pred_dist / pred_dist.sum()) /
                  (true_dist / true_dist.sum() + 1e-9) + 1e-9)
    print(f"Bias de classe estimado: {dict(zip(ID2LABEL.values(), np.round(bias, 2)))}")

    # --- Grid search alpha ---
    best_alpha, best_val_acc = 0.0, 0.0
    bias_t = torch.tensor(bias, dtype=torch.float32).unsqueeze(0)
    for alpha in np.arange(0.0, 2.6, 0.1):
        adj = val_logits - alpha * bias_t
        acc = accuracy_score(val_labels, adj.argmax(-1).numpy())
        if acc > best_val_acc:
            best_val_acc, best_alpha = acc, alpha
    print(f"Melhor alpha={best_alpha:.1f} -> val acc calibrado={best_val_acc:.4f}")

    # --- Logits do test com TTA ---
    test_logits_list, test_labels_list, test_paths = [], [], []
    for batch in tqdm(test_loader_tta, desc="TTA test"):
        # input_values: [B, n_crops, T]
        # attention_mask: [B, n_crops, T]
        B, N, T = batch["input_values"].shape
        iv_flat = batch["input_values"].view(B * N, T).to(DEVICE)
        am_flat = batch["attention_mask"].view(B * N, T).to(DEVICE)
        with torch.amp.autocast("cuda", enabled=False):
            logits_flat = model(iv_flat.float(), am_flat)   # [B*N, C]
        logits_avg = logits_flat.cpu().float().view(B, N, NUM_CLASSES).mean(dim=1)  # [B, C]
        test_logits_list.append(logits_avg)
        test_labels_list.extend(batch["labels"].numpy())
    test_logits = torch.cat(test_logits_list)
    test_labels = np.array(test_labels_list)

    # --- Aplicar correcao ---
    test_adj   = test_logits - best_alpha * bias_t
    test_preds = test_adj.argmax(-1).numpy()

    # --- Sem correcao ---
    raw_preds = test_logits.argmax(-1).numpy()

    return raw_preds, test_preds, test_labels


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # -- Carregar amostras --
    print("\n=== Carregando datasets ===")
    cafe_samples = get_cafe_samples(CAFE_DIR)
    asvp_samples = get_asvp_samples(ASVP_AUDIO, ASVP_BONUS)
    test_samples = get_test_samples(TEST_DIR)

    all_train = cafe_samples + asvp_samples
    if not all_train:
        raise RuntimeError("Nenhuma amostra de treino encontrada. Verifique DATA_ROOT.")

    print(f"\nDistribuicao de classes no treino:")
    cnt = Counter(l for _, l in all_train)
    for i, name in ID2LABEL.items():
        print(f"  {name:10s}: {cnt.get(i, 0):4d}")

    train_data, val_data = train_test_split(
        all_train, test_size=0.1, random_state=SEED,
        stratify=[l for _, l in all_train]
    )
    print(f"\nTreino: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_samples)}")

    # WeightedRandomSampler
    class_counts   = Counter(l for _, l in train_data)
    total          = sum(class_counts.values())
    sample_weights = [total / class_counts[l] for _, l in train_data]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # -- Feature extractor + modelo --
    print(f"\n=== Carregando backbone: {BACKBONE} ===")
    feature_extractor = AutoFeatureExtractor.from_pretrained(BACKBONE)
    model             = XLSREmotionClassifier(BACKBONE, NUM_CLASSES).to(DEVICE)

    n_total = sum(p.numel() for p in model.parameters())
    print(f"Parametros totais: {n_total/1e6:.1f}M")

    # -- DataLoaders --
    train_ds = SpeechEmotionDataset(train_data, feature_extractor, augment=True,  n_crops=1)
    val_ds   = SpeechEmotionDataset(val_data,   feature_extractor, augment=False, n_crops=1)
    test_ds  = SpeechEmotionDataset(test_samples, feature_extractor, augment=False, n_crops=TTA_CROPS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=4, shuffle=False,
                              num_workers=2, pin_memory=True)

    # -- Otimizador --
    encoder_params = [p for n, p in model.encoder.named_parameters() if p.requires_grad]
    head_params    = (list(model.layer_weights.unsqueeze(0))
                      + list(model.attn_proj.parameters())
                      + list(model.classifier.parameters()))
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(),  "lr": LR_ENCODER},
        {"params": model.layer_weights,          "lr": LR_HEAD},
        {"params": model.attn_proj.parameters(), "lr": LR_HEAD},
        {"params": model.classifier.parameters(),"lr": LR_HEAD},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = total_steps // 10

    from torch.optim.lr_scheduler import LambdaLR
    def lr_fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.02, 0.5 * (1.0 + np.cos(np.pi * progress)))
    scheduler = LambdaLR(optimizer, lr_fn)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler  = torch.amp.GradScaler("cuda")

    best_val_acc = 0.0
    os.makedirs(SAVE_DIR, exist_ok=True)

    # -- Resume de checkpoint --
    ckpt_path = os.path.join(SAVE_DIR, "model.pt")
    if RESUME_EPOCH > 0 and os.path.isfile(ckpt_path):
        print(f"\n=== Resumindo do checkpoint (epoch {RESUME_EPOCH}) ===")
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        best_val_acc = 0.689  # val acc do epoch 2 salvo
        # Avanca o scheduler para o ponto correto
        steps_done = RESUME_EPOCH * len(train_loader)
        for _ in range(steps_done):
            scheduler.step()
        print(f"  Scheduler avancado {steps_done} steps. Continuando do epoch {RESUME_EPOCH + 1}.")
    start_epoch = RESUME_EPOCH + 1

    # -- Loop de treino --
    print(f"\n=== Treinando epocas {start_epoch} a {EPOCHS} ===")
    for epoch in range(start_epoch, EPOCHS + 1):

        # Progressive unfreezing
        if epoch <= FREEZE_ALL:
            model.freeze_all()
            phase = f"CONGELADO (ep 1-{FREEZE_ALL})"
        elif epoch <= FREEZE_PARTIAL:
            model.unfreeze_last_n(UNFREEZE_LAYERS)
            phase = f"ultimas {UNFREEZE_LAYERS} camadas (ep {FREEZE_ALL+1}-{FREEZE_PARTIAL})"
        else:
            model.unfreeze_all()
            phase = "TUDO liberado"

        # Cabeca sempre treinavel
        for p in model.layer_weights.unsqueeze(0):
            pass  # layer_weights ja e Parameter, sempre treinavel
        for p in model.attn_proj.parameters():
            p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[Epoca {epoch}/{EPOCHS}] {phase} | Parametros treinaveis: {trainable/1e6:.1f}M")

        # fp16 so quando encoder esta congelado (seguro e rapido com 0.5M params).
        # A partir da fase 2 (101M params + transformer layers), fp32 e obrigatorio:
        # GradScaler detecta Inf mas NAO NaN nos gradientes -- LayerNorm com ativacoes
        # grandes em fp16 produz NaN silencioso que corrompe pesos permanentemente.
        # RTX 4090 (24GB): modelo fp32 ~5GB total, perfeitamente viavel.
        use_amp = (epoch <= FREEZE_ALL)

        model.train()
        total_loss    = 0.0
        n_valid       = 0
        n_nan_logits  = 0
        n_nan_grad    = 0
        pbar = tqdm(train_loader, desc=f"  Treino", leave=False)
        for batch in pbar:
            iv  = batch["input_values"].to(DEVICE)
            am  = batch["attention_mask"].to(DEVICE)
            lbl = batch["labels"].to(DEVICE)
            optimizer.zero_grad()

            # Forward (fp16 ou fp32 dependendo da fase)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(iv if use_amp else iv.float(), am)

            # Guarda NaN/Inf nos logits antes de qualquer coisa
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                n_nan_logits += 1
                pbar.set_postfix(loss="NaN-logit")
                continue

            loss = loss_fn(logits.float(), lbl)
            if torch.isnan(loss) or torch.isinf(loss):
                n_nan_logits += 1
                pbar.set_postfix(loss="NaN-loss")
                continue

            # Backward
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    optimizer.zero_grad()
                    scaler.update()   # obrigatorio chamar mesmo pulando
                    n_nan_grad += 1
                    pbar.set_postfix(loss="NaN-grad")
                    continue
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    optimizer.zero_grad()
                    n_nan_grad += 1
                    pbar.set_postfix(loss="NaN-grad")
                    continue
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()
            n_valid    += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(1, n_valid)
        if n_nan_logits or n_nan_grad:
            print(f"  [NaN] {n_nan_logits} logit-NaN, {n_nan_grad} grad-NaN pulados de {len(train_loader)} batches")

        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader)
        print(f"  Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Recovery: se colapso total (NaN corrompe pesos), recarrega ultimo bom checkpoint
        if val_acc == 0.0 and best_val_acc > 0.0 and os.path.isfile(ckpt_path):
            print(f"  [RECOVERY] Colapso detectado -- recarregando checkpoint (best={best_val_acc:.4f})")
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
            continue

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model.pt"))
            feature_extractor.save_pretrained(SAVE_DIR)
            print(f"  -> Checkpoint salvo (val_acc={val_acc:.4f})")

    print(f"\n=== Melhor val acc: {best_val_acc:.4f} ===")

    # -- Avaliacao no teste --
    print("\n=== Carregando melhor checkpoint para avaliacao ===")
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model.pt"), map_location=DEVICE, weights_only=True))

    print("\n=== Calibracao de prior + TTA ===")
    raw_preds, cal_preds, test_labels_arr = calibrate_and_predict(
        model, val_loader, test_loader
    )

    print(f"\nAcuracia sem calibracao: {accuracy_score(test_labels_arr, raw_preds):.4f}")
    print(f"Acuracia com calibracao:  {accuracy_score(test_labels_arr, cal_preds):.4f}")
    print("\nRelatorio de classificacao (com calibracao):")
    print(classification_report(
        test_labels_arr, cal_preds,
        target_names=[ID2LABEL[i] for i in range(NUM_CLASSES)]
    ))
    print(f"Distribuicao predita: {Counter(ID2LABEL[p] for p in cal_preds)}")

    # Salvar resultados
    results_path = os.path.join(os.path.dirname(SAVE_DIR), "test_results_v2.csv")
    pd.DataFrame({
        "file":       [s[0] for s in test_samples],
        "true_label": [ID2LABEL[l] for l in test_labels_arr],
        "pred_label": [ID2LABEL[p] for p in cal_preds],
    }).to_csv(results_path, index=False)
    print(f"\nResultados salvos em: {results_path}")


if __name__ == "__main__":
    main()
