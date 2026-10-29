#!/usr/bin/env bash
# Downloads MOT17 and keeps ONLY sequence MOT17-09 (stationary camera, 525 frames),
# then copies N evenly spaced frames into data/mot17_subset/ for Layer 1 validation.
# The full sequence (data/MOT17/) is also used as Layer 2 tracking input.
# MOT17 is licensed CC BY-NC-SA 3.0 (motchallenge.net), so it is fetched, not committed.
set -euo pipefail
cd "$(dirname "$0")"
SEQ=MOT17-09-FRCNN      # all MOT17 detector variants share the same images
N=${1:-40}

if [ ! -d "MOT17/train/$SEQ/img1" ]; then
  [ -f MOT17.zip ] || wget -O MOT17.zip https://motchallenge.net/data/MOT17.zip
  unzip -q MOT17.zip "MOT17/train/$SEQ/*"
  rm -f MOT17.zip
fi

mkdir -p mot17_subset
rm -f mot17_subset/*.jpg
TOTAL=$(ls "MOT17/train/$SEQ/img1" | wc -l)
STEP=$(( TOTAL / N ))
ls "MOT17/train/$SEQ/img1" | sort | awk -v s="$STEP" 'NR % s == 1' | head -n "$N" \
  | while read f; do cp "MOT17/train/$SEQ/img1/$f" mot17_subset/; done
echo "Sequence: $TOTAL frames in data/MOT17/train/$SEQ/img1"
echo "Subset:   $(ls mot17_subset | wc -l) frames in data/mot17_subset/ (every ${STEP}th)"
