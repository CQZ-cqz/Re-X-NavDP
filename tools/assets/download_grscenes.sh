#!/usr/bin/env bash
# Download GRScenes-100 (the actual train/eval scene USD) from HF.
# GRScenes is public (not gated) and contains all 99 scenes: 69 home + 30 commercial.
set -uo pipefail

export HF_ENDPOINT=https://hf-mirror.com
DEST="${GRSCENES_DEST:-/mnt/data3/cqz/nav/grscenes}"
PY=huggingface-cli
mkdir -p "$DEST"

for f in \
    scenes/GRScenes-100/home_scenes.zip \
    scenes/GRScenes-100/home_scenes.z01 \
    scenes/GRScenes-100/home_scenes.z02 \
    scenes/GRScenes-100/commercial_scenes.zip; do
    base="$(basename "$f")"
    if [ -f "$DEST/$base" ]; then
        echo "[skip] $base already complete"
        continue
    fi
    for i in $(seq 1 20); do
        echo "[download] $base attempt $i  $(date '+%F %T')"
        if "$PY" download InternRobotics/GRScenes "$f" --repo-type dataset --local-dir "$DEST"; then
            echo "[done] $base"
            break
        else
            echo "[retry] $base attempt $i failed; sleeping 30s"
            sleep 30
        fi
    done
done
echo "ALL DONE $(date '+%F %T')"
