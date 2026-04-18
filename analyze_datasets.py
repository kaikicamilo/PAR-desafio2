# Analise completa dos datasets para verificar alinhamento de labels.


import os, glob, unicodedata, re
from pathlib import Path
from collections import Counter, defaultdict

DATA_ROOT  = "/workspace/kaiki_home/data"
CAFE_DIR   = os.path.join(DATA_ROOT, "anad_cafe")
ASVP_AUDIO = os.path.join(DATA_ROOT, "asvp_esd", "ASVP_UPDATE", "Audio")
ASVP_BONUS = os.path.join(DATA_ROOT, "asvp_esd", "ASVP_UPDATE", "Bonus")
TEST_DIR   = os.path.join(DATA_ROOT, "test_set", "SUBESCO")

ID2LABEL = {0:"angry", 1:"disgust", 2:"fear", 3:"happy",
            4:"neutral", 5:"sad", 6:"surprise"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# Mapeamento CaFE (normalizado)
def _norm(s):
    return unicodedata.normalize("NFD", s).encode("ascii","ignore").decode().lower().strip()

CAFE_MAP_NORM = {
    "colere":    0,
    "degout":    1,
    "degoat":    1,
    "peur":      2,
    "joie":      3,
    "neutre":    4,
    "tristesse": 5,
    "surprise":  6,
}

# Mapeamento RAVDESS/ASVP (codigo -> label id)
RAVDESS_MAP = {
    "01": (4, "neutral"),
    "02": (None, "calm -- EXCLUIDO"),
    "03": (3, "happy"),
    "04": (5, "sad"),
    "05": (0, "angry"),
    "06": (2, "fearful"),
    "07": (1, "disgust"),
    "08": (6, "surprised"),
}

SEP = "=" * 70


def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def check_dir(path, name):
    exists = os.path.isdir(path)
    status = "OK" if exists else "NAO ENCONTRADO"
    print(f"  [{status}] {name}: {path}")
    return exists

section("1. ESTRUTURA DE PASTAS")

print("\nRaiz dos dados:")
check_dir(DATA_ROOT, "DATA_ROOT")

print("\nDatasets de treino:")
check_dir(CAFE_DIR,   "CaFE      (anad_cafe)")
check_dir(ASVP_AUDIO, "ASVP_ESD  (Audio)")
check_dir(ASVP_BONUS, "ASVP_ESD  (Bonus)")

print("\nDataset de teste:")
check_dir(TEST_DIR, "SUBESCO   (test_set/SUBESCO)")

# Listar estrutura de nivel 1 e 2
if os.path.isdir(DATA_ROOT):
    print("\nConteudo de DATA_ROOT (ate nivel 2):")
    for item in sorted(os.listdir(DATA_ROOT)):
        full = os.path.join(DATA_ROOT, item)
        if os.path.isdir(full):
            sub_items = sorted(os.listdir(full))[:8]
            n_all = len(os.listdir(full))
            print(f"  {item}/  ({n_all} itens)")
            for s in sub_items:
                print(f"    {s}")
            if n_all > 8:
                print(f"    ... +{n_all-8} mais")


section("2. CaFE (anad_cafe) -- Analise de Labels")

cafe_label_counts = Counter()
cafe_samples = []
cafe_unmapped_folders = []
cafe_per_folder = {}

if os.path.isdir(CAFE_DIR):
    folders = sorted(os.listdir(CAFE_DIR))
    print(f"\nPastas encontradas em anad_cafe/ ({len(folders)} total):")
    for d in folders:
        fpath = os.path.join(CAFE_DIR, d)
        if not os.path.isdir(fpath):
            print(f"  [ARQUIVO] {d}  (ignorado)")
            continue
        normed = _norm(d)
        label_id = CAFE_MAP_NORM.get(normed)
        label_name = ID2LABEL.get(label_id, "???")
        wavs = glob.glob(os.path.join(fpath, "**", "*.wav"), recursive=True)
        n = len(wavs)

        if label_id is None:
            status = "SEM MAPEAMENTO"
            cafe_unmapped_folders.append(d)
        else:
            status = f"-> {label_name} (id={label_id})"
            cafe_label_counts[label_id] += n
            for w in wavs:
                cafe_samples.append((w, label_id))

        print(f"  '{d}'  [norm='{normed}']  {n:4d} wavs  {status}")
        cafe_per_folder[d] = n

        # Listar subpastas (intensidade, locutor, etc.)
        subs = [s for s in sorted(os.listdir(fpath)) if os.path.isdir(os.path.join(fpath, s))]
        if subs:
            print(f"    Subpastas: {subs[:10]}" + (" ..." if len(subs)>10 else ""))

    print(f"\nPastas SEM mapeamento: {cafe_unmapped_folders}")

    print(f"\nContagem de amostras CaFE por label (MAPEADAS):")
    for lid, lname in ID2LABEL.items():
        n = cafe_label_counts.get(lid, 0)
        bar = "#" * (n // 10)
        print(f"  {lname:10s} (id={lid}): {n:5d}  {bar}")
    print(f"  TOTAL: {len(cafe_samples)}")

    # Mostrar exemplos de nomes de arquivo
    print(f"\nExemplos de nomes de arquivo CaFE (5 por emocao):")
    samples_by_label = defaultdict(list)
    for path, lbl in cafe_samples:
        samples_by_label[lbl].append(Path(path).name)
    for lid in sorted(samples_by_label):
        examples = samples_by_label[lid][:5]
        print(f"  {ID2LABEL[lid]:10s}: {examples}")
else:
    print("  [AVISO] Diretorio CaFE nao encontrado.")

section("3. ASVP_ESD -- Analise de Labels por Nome de Arquivo")

asvp_label_counts  = Counter()
asvp_code_counts   = Counter()
asvp_unmapped      = []
asvp_samples       = []
asvp_malformed     = []

for search_dir, dirname in [(ASVP_AUDIO, "Audio"), (ASVP_BONUS, "Bonus")]:
    if not os.path.isdir(search_dir):
        print(f"\n  [NAO ENCONTRADO] {dirname}: {search_dir}")
        continue

    wavs = glob.glob(os.path.join(search_dir, "**", "*.wav"), recursive=True)
    print(f"\nDiretorio '{dirname}': {len(wavs)} arquivos .wav")

    # Mostrar estrutura de subpastas
    subdirs = [d for d in sorted(os.listdir(search_dir))
               if os.path.isdir(os.path.join(search_dir, d))]
    if subdirs:
        print(f"  Subpastas: {subdirs[:10]}" + (" ..." if len(subdirs)>10 else ""))

    # Analise dos nomes
    code_samples = defaultdict(list)
    for fpath in wavs:
        stem = Path(fpath).stem
        parts = stem.split("-")
        if len(parts) < 3:
            asvp_malformed.append(stem)
            continue
        code = parts[2]
        asvp_code_counts[code] += 1
        info = RAVDESS_MAP.get(code)
        if info is None:
            asvp_unmapped.append(stem)
        else:
            label_id, label_name = info
            if label_id is not None:
                asvp_label_counts[label_id] += 1
                asvp_samples.append((fpath, label_id))
            code_samples[code].append(Path(fpath).name)

    print(f"\n  Distribuicao de CODIGOS de emocao encontrados:")
    for code in sorted(asvp_code_counts):
        info = RAVDESS_MAP.get(code, (None, "DESCONHECIDO"))
        lid, lname = info
        status = f"-> {lname} (id={lid})" if lid is not None else f"-> {lname}"
        n = asvp_code_counts[code]
        bar = "#" * (n // 20)
        print(f"    codigo '{code}': {n:5d}  {status}  {bar}")
        # Exemplos
        examples = code_samples.get(code, [])[:3]
        if examples:
            print(f"      ex: {examples}")

print(f"\nArquivos malformados (menos de 3 campos): {len(asvp_malformed)}")
if asvp_malformed:
    print(f"  Exemplos: {asvp_malformed[:5]}")

print(f"\nContagem de amostras ASVP_ESD por label (MAPEADAS, sem calm):")
for lid, lname in ID2LABEL.items():
    n = asvp_label_counts.get(lid, 0)
    bar = "#" * (n // 20)
    print(f"  {lname:10s} (id={lid}): {n:5d}  {bar}")
print(f"  TOTAL: {len(asvp_samples)}")


section("4. SUBESCO (test set) -- Analise de Labels por Nome de Arquivo")

test_label_counts = Counter()
test_samples      = []
test_unmatched    = []

if os.path.isdir(TEST_DIR):
    all_wavs = glob.glob(os.path.join(TEST_DIR, "*.wav"))
    print(f"\nArquivos .wav encontrados: {len(all_wavs)}")

    # Mostrar exemplos de nomes
    print("\nPrimeiros 20 nomes de arquivo:")
    for w in sorted(all_wavs)[:20]:
        print(f"  {Path(w).name}")

    # Analise de labels por nome
    print("\nExtracao de label pelo nome:")
    for fpath in all_wavs:
        name = Path(fpath).stem.upper()
        matched = False
        for emo_str, emo_id in LABEL2ID.items():
            if f"_{emo_str.upper()}_" in name or name.endswith(f"_{emo_str.upper()}"):
                test_label_counts[emo_id] += 1
                test_samples.append((fpath, emo_id))
                matched = True
                break
        if not matched:
            test_unmatched.append(Path(fpath).name)

    print(f"\nContagem por label:")
    for lid, lname in ID2LABEL.items():
        n = test_label_counts.get(lid, 0)
        bar = "#" * (n // 50)
        print(f"  {lname:10s} (id={lid}): {n:5d}  {bar}")
    print(f"  TOTAL mapeado: {len(test_samples)}")
    print(f"  NAO mapeados:  {len(test_unmatched)}")

    if test_unmatched:
        print(f"\nExemplos NAO mapeados (primeiros 10):")
        for nm in test_unmatched[:10]:
            print(f"  {nm}")

    # Verificar padroes do nome para entender a convencao
    print("\nAnalise do padrao de nome (primeiros 5 por emocao):")
    by_label = defaultdict(list)
    for fpath, lid in test_samples:
        by_label[lid].append(Path(fpath).name)
    for lid in sorted(by_label):
        examples = sorted(by_label[lid])[:5]
        print(f"  {ID2LABEL[lid]:10s}: {examples}")
else:
    print("  [AVISO] Diretorio SUBESCO nao encontrado.")

section("5. ALINHAMENTO DE LABELS (resumo)")

print("\nMapeamento definitivo usado no treinamento:")
print(f"  {'Label ID':8s} {'Nome':12s} {'CaFE':>8s} {'ASVP':>8s} {'SUBESCO':>8s} {'TOTAL_TREINO':>12s}")
print(f"  {'-'*60}")
total_train = 0
for lid, lname in ID2LABEL.items():
    n_cafe   = cafe_label_counts.get(lid, 0)
    n_asvp   = asvp_label_counts.get(lid, 0)
    n_test   = test_label_counts.get(lid, 0)
    n_train  = n_cafe + n_asvp
    total_train += n_train
    imbalance = "" if n_train > 200 else "  !! POUCOS DADOS"
    print(f"  {lid:<8d} {lname:<12s} {n_cafe:>8d} {n_asvp:>8d} {n_test:>8d} {n_train:>12d}{imbalance}")
print(f"  {'-'*60}")
print(f"  {'TOTAL':>20s} {sum(cafe_label_counts.values()):>8d} "
      f"{sum(asvp_label_counts.values()):>8d} "
      f"{sum(test_label_counts.values()):>8d} "
      f"{total_train:>12d}")

# Verificar se alguma label do test nao aparece no treino
print("\nLabels presentes no teste mas ausentes/escassas no treino:")
for lid, lname in ID2LABEL.items():
    n_train = cafe_label_counts.get(lid, 0) + asvp_label_counts.get(lid, 0)
    n_test  = test_label_counts.get(lid, 0)
    if n_test > 0 and n_train == 0:
        print(f"  [CRITICO] {lname} (id={lid}): 0 amostras de treino, {n_test} de teste!")
    elif n_test > 0 and n_train < 100:
        print(f"  [AVISO]   {lname} (id={lid}): apenas {n_train} amostras de treino, {n_test} de teste")

# Verificar consistencia de IDs (se alguma emocao tem ID diferente entre datasets)
print("\nConsistencia do mapeamento de IDs:")
print("  CaFE mapeamento:")
for k, v in CAFE_MAP_NORM.items():
    print(f"    pasta_norm='{k}' -> id={v} ({ID2LABEL.get(v,'???')})")
print("  ASVP/RAVDESS mapeamento:")
for code, (lid, lname) in RAVDESS_MAP.items():
    mapped = ID2LABEL.get(lid, "EXCLUIDO") if lid is not None else "EXCLUIDO"
    print(f"    codigo='{code}' label_original='{lname}' -> id={lid} ({mapped})")


section("6. VERIFICACAO DE QUALIDADE (amostra de arquivos)")

import wave, struct

def check_wav(fpath):
    """Retorna (duration_s, sample_rate, channels) ou None se corrompido."""
    try:
        with wave.open(fpath, "rb") as wf:
            sr   = wf.getframerate()
            ch   = wf.getnchannels()
            nf   = wf.getnframes()
            dur  = nf / sr
            return dur, sr, ch
    except Exception as e:
        return None

def check_sample(samples, name, n=30):
    print(f"\nVerificando {n} arquivos aleatorios de {name}...")
    import random
    rng = random.Random(42)
    subset = rng.sample(samples, min(n, len(samples)))
    errors = 0
    sr_counter = Counter()
    ch_counter = Counter()
    short = 0
    for fpath, label in subset:
        result = check_wav(fpath)
        if result is None:
            print(f"  [ERRO] {Path(fpath).name}")
            errors += 1
            continue
        dur, sr, ch = result
        sr_counter[sr] += 1
        ch_counter[ch] += 1
        if dur < 0.1:
            short += 1
    print(f"  Erros/corrompidos: {errors}/{len(subset)}")
    print(f"  Sample rates: {dict(sr_counter)}")
    print(f"  Canais: {dict(ch_counter)}")
    print(f"  Muito curtos (<0.1s): {short}")

if cafe_samples:
    check_sample(cafe_samples, "CaFE")
if asvp_samples:
    check_sample(asvp_samples, "ASVP_ESD")
if test_samples:
    check_sample(test_samples, "SUBESCO")


section("7. DIAGNOSTICO FINAL")

issues = []
warnings_list = []

# Verifica pastas nao mapeadas no CaFE
if cafe_unmapped_folders:
    issues.append(f"CaFE: {len(cafe_unmapped_folders)} pastas sem mapeamento: {cafe_unmapped_folders}")

# Verifica classes com 0 amostras de treino mas presentes no teste
for lid in ID2LABEL:
    n_train = cafe_label_counts.get(lid,0) + asvp_label_counts.get(lid,0)
    n_test  = test_label_counts.get(lid,0)
    if n_test > 0 and n_train == 0:
        issues.append(f"Label '{ID2LABEL[lid]}' (id={lid}): ZERO amostras de treino mas {n_test} no teste!")

# Verifica desbalanceamento extremo
for lid in ID2LABEL:
    n_train = cafe_label_counts.get(lid,0) + asvp_label_counts.get(lid,0)
    if 0 < n_train < 100:
        warnings_list.append(f"Label '{ID2LABEL[lid]}' (id={lid}): apenas {n_train} amostras de treino (muito pouco)")

# Verifica SUBESCO nao mapeado
if test_unmatched:
    if len(test_unmatched) > len(test_samples) * 0.05:
        issues.append(f"SUBESCO: {len(test_unmatched)} arquivos ({100*len(test_unmatched)/(len(test_unmatched)+len(test_samples)):.1f}%) nao mapeados para label")
    else:
        warnings_list.append(f"SUBESCO: {len(test_unmatched)} arquivos nao mapeados (<5%, aceitavel)")

if issues:
    print("\n[CRITICO] Problemas que DEVEM ser corrigidos antes do treino:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("\n[OK] Nenhum problema critico encontrado.")

if warnings_list:
    print("\n[AVISO] Pontos de atencao:")
    for i, w in enumerate(warnings_list, 1):
        print(f"  {i}. {w}")

print(f"\nResumo de amostras:")
print(f"  Treino: {len(cafe_samples) + len(asvp_samples)} (CaFE={len(cafe_samples)}, ASVP={len(asvp_samples)})")
print(f"  Teste:  {len(test_samples)} (SUBESCO mapeados)")
print(f"\nAnalise concluida.")
