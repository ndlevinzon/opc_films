#!/bin/bash
# Representative hold frames for main text Fig. 5c, colored by the display
# classifier. Frames go to results/lda_frames; the colored cross sections to
# results/lda_frames/phase_density_maps.
set -euo pipefail
cd "$(dirname "$0")/../.."
repo=$(pwd)
if [ -z "${OPC_DATA_ROOT:-}" ]; then
  top=$(python -c "import pandas as pd; m=pd.read_csv('config/manifest.tsv', sep='\t', dtype=str, keep_default_na=False); print(next(value for value in m.topology if value))")
  export OPC_DATA_ROOT=$(dirname "$(dirname "$top")")
fi
export OPC_MANIFEST="$repo/config/manifest.tsv"
export OPC_FRAMES_OUT="$repo/results/lda_frames"
mkdir -p "$OPC_FRAMES_OUT"
# Extraction is slow; frames from an earlier attempt are reused when every
# hold has one. Delete results/lda_frames to force a new selection.
holds=$(python -c "import pandas as pd; m=pd.read_csv('config/manifest.tsv', sep='\t', dtype=str, keep_default_na=False); print((m.kind=='hold').sum())")
have=$(find "$OPC_FRAMES_OUT" -maxdepth 1 -name 'T*_r*_frame*.pdb' | wc -l)
if [ "$have" -lt "$holds" ]; then
  python analysis/glass_structure/render_nvt_temp_x_sections/extract_vitrification_frames.py
else
  echo "Reusing $have extracted frames in $OPC_FRAMES_OUT"
fi
python analysis/glass_structure/render_nvt_temp_x_sections/plot_color_LDA.py --input-dir "$OPC_FRAMES_OUT"
