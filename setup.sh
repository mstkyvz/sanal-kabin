#!/bin/bash
# Kurulum: upstream repoları sabit commit'ten indirir, yamaları uygular, ortamı kurar, modelleri indirir.
set -euo pipefail
cd "$(dirname "$0")"

CATVTON_COMMIT=7818397f25613beedb3d861a34769f607cfcf3b1
DMVTON_COMMIT=5dcd04ee3f5b1fb392df1184c887c1a7e910c72e

fetch() { # repo dir commit patch
  if [ ! -d "$2" ]; then
    git clone https://github.com/$1.git "$2"
    git -C "$2" checkout -q "$3"
    git -C "$2" apply "$PWD/$4"
  fi
}
fetch Zheng-Chong/CatVTON CatVTON $CATVTON_COMMIT patches/catvton-mac.patch
fetch KiseKloset/DM-VTON DM-VTON $DMVTON_COMMIT patches/dmvton-mac.patch

if command -v uv >/dev/null; then
  [ -d .venv ] || uv venv --python 3.11 .venv
  source .venv/bin/activate
  uv pip install -r requirements.txt
else
  [ -d .venv ] || python3.11 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
fi

python scripts/download_models.py
echo "Hazır. Başlatmak için: ./start.sh"
