#!/usr/bin/env bash
# Merge GRScenes-100 (69 home + 30 commercial) into the assembled SCENE_DIR.
# Uses --ignore-existing so the already-validated eval scenes (from n1_eval_scenes)
# are kept, and only the missing TRAIN scenes + missing models/Materials are added.
set -euo pipefail

EXT=/mnt/data3/cqz/nav/grscenes/extracted
SCENES=/mnt/data3/cqz/nav/scenes

merge_one() {
    local src="$1" dst="$2"
    if [ -d "$src" ]; then
        rsync -a --ignore-existing "$src/" "$dst/"
        echo "[merged] $src -> $dst"
    else
        echo "[skip] $src (not found)"
    fi
}

# Commercial: models, Materials, scenes -> scenes_commercial
merge_one "$EXT/commercial_scenes/models"     "$SCENES/internscenes_commercial/models"
merge_one "$EXT/commercial_scenes/Materials"  "$SCENES/internscenes_commercial/Materials"
merge_one "$EXT/commercial_scenes/scenes"     "$SCENES/internscenes_commercial/scenes_commercial"

# Home: models, Materials, scenes -> scenes_home
merge_one "$EXT/home_scenes/models"    "$SCENES/internscenes_home/models"
merge_one "$EXT/home_scenes/Materials" "$SCENES/internscenes_home/Materials"
merge_one "$EXT/home_scenes/scenes"    "$SCENES/internscenes_home/scenes_home"

echo "=== 合并后场景数 ==="
echo "home:      $(ls -1 "$SCENES/internscenes_home/scenes_home" | wc -l) (应 69)"
echo "commercial:$(ls -1 "$SCENES/internscenes_commercial/scenes_commercial" | wc -l) (应 30)"
echo "ALL DONE $(date '+%F %T')"
