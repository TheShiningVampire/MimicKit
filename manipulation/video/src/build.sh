#!/usr/bin/env bash
# Build the explainer video: TTS -> manim scenes (with embedded narration) -> concat.
# Usage: bash build.sh [quality]   quality: l (480p15, fast) | h (1080p30, default)
set -euo pipefail
cd "$(dirname "$0")"
Q=${1:-h}
MEDIA=/data/user_data/vawale/MimicKit/manip_video/media
AUDIO=/data/user_data/vawale/MimicKit/manip_video/audio
OUT=/data/user_data/vawale/MimicKit/manip_video/manipulation_explained.mp4
TTS_PY=/data/user_data/vawale/envs/tts/bin/python
MANIM=/data/user_data/vawale/envs/manim/bin/manim
SCENES="S01_Title S02_Recap S03_WhatChanges S04_Contacts S05_Observation S06_Data S07_Retarget S08_Pipeline S09_Plan S10_Close"

[ -f $AUDIO/timing.json ] || $TTS_PY gen_audio.py --out $AUDIO
if [ "$Q" = "l" ]; then FLAGS="-ql"; DIR=480p15; else FLAGS="-r 1920,1080 --fps 30"; DIR=1080p30; fi
for s in $SCENES; do
  NARRATION_DIR=$AUDIO $MANIM $FLAGS --disable_caching --media_dir $MEDIA scenes.py $s
done
: > $MEDIA/concat.txt
for s in $SCENES; do echo "file '$MEDIA/videos/scenes/$DIR/$s.mp4'" >> $MEDIA/concat.txt; done
ffmpeg -loglevel error -y -f concat -safe 0 -i $MEDIA/concat.txt -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -c:a aac -b:a 160k -movflags +faststart $OUT
echo "wrote $OUT"; ffprobe -v error -show_entries format=duration -of csv=p=0 $OUT
