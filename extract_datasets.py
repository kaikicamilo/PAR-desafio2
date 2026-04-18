# Extrai os zips dos datasets com tratamento correto de encoding.


import zipfile
import os
import shutil
import sys

CAFE_ZIP  = "cafe.zip"     
ASVP_ZIP  = "asvp.zip"      
TEST_ZIP  = "SUBESCO.zip"        
OUT_DIR   = "/workspace/kaiki_home/data" 


def fix_name(raw: str) -> str:
    """Tenta corrigir encoding de nomes corrompidos pelo unzip padrao."""
    for enc in ("utf-8", "cp850", "cp437", "latin-1", "cp1252"):
        try:
            return raw.encode("cp437").decode(enc)
        except Exception:
            continue
    return raw  # fallback: usa o nome como esta


def extract_zip(zip_path: str, dest: str):
    """Extrai zip para dest com correcao de encoding nos nomes."""
    if not os.path.isfile(zip_path):
        print(f"[AVISO] Arquivo nao encontrado: {zip_path} -- pulando")
        return

    print(f"\nExtraindo {zip_path} -> {dest}")
    os.makedirs(dest, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            # Corrige nome do arquivo/pasta
            if info.flag_bits & 0x800:
                # Flag UTF-8 ativa -- nome ja esta correto
                name = info.filename
            else:
                # Encoding legado (CP437 do Windows)
                name = fix_name(info.filename)

            target = os.path.join(dest, name)

            if info.is_dir():
                os.makedirs(target, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    print(f"  Concluido.")


def check_structure(data_root: str):
    """Mostra o que foi extraido para confirmar."""
    print("\n=== Estrutura extraida ===")
    for dirpath, dirnames, filenames in os.walk(data_root):
        depth = dirpath.replace(data_root, "").count(os.sep)
        if depth > 3:
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(dirpath)}/")
        if depth == 2:
            wav_count = sum(1 for f in filenames if f.endswith(".wav"))
            if wav_count:
                print(f"{indent}  [{wav_count} arquivos .wav]")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    extract_zip(CAFE_ZIP, OUT_DIR)
    extract_zip(ASVP_ZIP, OUT_DIR)
    extract_zip(TEST_ZIP, OUT_DIR)

    check_structure(OUT_DIR)

    print("\n=== Pastas encontradas em anad_cafe ===")
    cafe_path = os.path.join(OUT_DIR, "anad_cafe")
    if os.path.isdir(cafe_path):
        for d in sorted(os.listdir(cafe_path)):
            full = os.path.join(cafe_path, d)
            n_wav = sum(1 for _, _, fs in os.walk(full) for f in fs if f.endswith(".wav"))
            print(f"  '{d}'  ({n_wav} wavs)")
    else:
        print(f"  [AVISO] Pasta nao encontrada: {cafe_path}")
        print("  Verifique o nome da pasta raiz dentro do zip.")
