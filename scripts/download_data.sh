#!/usr/bin/env bash
# Downloads RAM-H1200 (gated). First accept the terms at
# https://huggingface.co/datasets/TokyoTechMagicYang/RAM-H1200-v1 and run `huggingface-cli login`.
set -euo pipefail
DEST="${1:-data/RAM-H1200-v1}"
mkdir -p "$DEST"
huggingface-cli download TokyoTechMagicYang/RAM-H1200-v1 --repo-type dataset --local-dir "$DEST"
echo "Downloaded to $DEST. Next: make inspect"
