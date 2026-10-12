#!/usr/bin/env bash
# Downloads RAM-H1200 (gated). First accept the terms at
# https://huggingface.co/datasets/TokyoTechMagicYang/RAM-H1200-v1 while logged in,
# then log in here once:  python -c "from huggingface_hub import login; login()"
set -euo pipefail
DEST="${1:-data/RAM-H1200-v1}"
mkdir -p "$DEST"
python -c "from huggingface_hub import snapshot_download; snapshot_download('TokyoTechMagicYang/RAM-H1200-v1', repo_type='dataset', local_dir='$DEST')"
echo "Downloaded to $DEST. Next: make inspect"
