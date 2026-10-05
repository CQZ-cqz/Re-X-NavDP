#!/usr/bin/env bash
# Extract GRScenes-100 (69 home + 30 commercial scenes) from the downloaded archives.
set -uo pipefail

SRC=/mnt/data3/cqz/nav/grscenes/scenes/GRScenes-100
DST=/mnt/data3/cqz/nav/grscenes/extracted
mkdir -p "$DST"
cd "$DST"

echo "[extract] home_scenes (split zip) $(date '+%F %T')"
unzip -o -q "$SRC/home_scenes.zip" && echo "[done] home_scenes"

echo "[extract] commercial_scenes $(date '+%F %T')"
unzip -o -q "$SRC/commercial_scenes.zip" && echo "[done] commercial_scenes"

echo "ALL DONE $(date '+%F %T')"
