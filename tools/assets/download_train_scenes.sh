#!/usr/bin/env bash
# Download the gated train-scene archives (mp3d_*) with retry + resume.
set -uo pipefail

export HF_ENDPOINT=https://hf-mirror.com
DEST="${TRAIN_SCENE_DEST:-/mnt/data3/cqz/nav/train_scenes}"
PY=huggingface-cli
mkdir -p "$DEST"

for f in mp3d_ce.tar.gz mp3d_n1.tar.gz mp3d_pe.tar.gz; do
    if [ -f "$DEST/$f" ]; then
        echo "[skip] $f already complete"
        continue
    fi
    for i in $(seq 1 20); do
        echo "[download] $f attempt $i  $(date '+%F %T')"
        if "$PY" download InternRobotics/Scene-N1 "$f" --repo-type dataset --local-dir "$DEST"; then
            echo "[done] $f"
            break
        else
            echo "[retry] $f attempt $i failed; sleeping 30s"
            sleep 30
        fi
    done
done
echo "ALL DONE $(date '+%F %T')"
