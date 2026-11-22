#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# add_custom_recordings.sh
#
# Mixes real recordings of you saying "Hey Aaryan" into the training set.
# The default pipeline only trains on synthetic Piper TTS voices, which
# don't capture your actual voice, accent, mic, or room acoustics - adding
# even 20-50 real clips noticeably improves real-world detection confidence.
#
# Usage:
#   1. Record yourself saying "Hey Aaryan" ~30-50+ times, in the room and at
#      the mic distance you'll actually use it from. Vary your tone/pace a
#      little across takes. Save each take as its own short audio file
#      (wav/mp3/m4a/ogg/flac - any format/sample rate, this script converts
#      them). Windows' Voice Recorder app or your phone's voice memo app
#      both work fine. Keep each clip to just the phrase itself, roughly
#      1-2 seconds, not a long recording with pauses or multiple takes.
#   2. Copy all the files into one folder, e.g.
#      ~/wakeword-training/my_recordings/  (from Windows, WSL2's filesystem
#      is browseable at \\wsl$\Ubuntu\home\<you>\... in Explorer)
#   3. From inside $WORKDIR/openWakeWord, with the oww-env venv active, run:
#      bash add_custom_recordings.sh ~/wakeword-training/my_recordings
#   4. Continue (or re-run) the pipeline as normal. If you've already run
#      --generate_clips and --augment_clips before adding these, re-run
#      just the augmentation + training steps with --overwrite so the new
#      clips are picked up:
#        python3 openwakeword/train.py --training_config hey_aaryan.yml \
#          --augment_clips --overwrite
#        python3 openwakeword/train.py --training_config hey_aaryan.yml \
#          --train_model
#      ...then the onnx2tf conversion step (see run_training.sh Step 4, or
#      just re-run run_training.sh directly - --generate_clips and
#      --augment_clips are skipped automatically once enough clips/features
#      already exist, so re-running is safe and fast except for training).
# ---------------------------------------------------------------------------
set -e

SRC_DIR="$1"
MODEL_NAME="hey_aaryan"
MODEL_DIR="my_custom_model"
POS_TRAIN="$MODEL_DIR/$MODEL_NAME/positive_train"
POS_TEST="$MODEL_DIR/$MODEL_NAME/positive_test"

if [ -z "$SRC_DIR" ] || [ ! -d "$SRC_DIR" ]; then
  echo "Usage: bash add_custom_recordings.sh <folder-of-recordings>"
  exit 1
fi

mkdir -p "$POS_TRAIN" "$POS_TEST"

shopt -s nullglob nocaseglob
files=("$SRC_DIR"/*.wav "$SRC_DIR"/*.mp3 "$SRC_DIR"/*.m4a "$SRC_DIR"/*.ogg "$SRC_DIR"/*.flac)
shopt -u nocaseglob

if [ ${#files[@]} -eq 0 ]; then
  echo "No audio files found in $SRC_DIR"
  exit 1
fi

echo ">>> Converting ${#files[@]} recordings in $SRC_DIR to 16kHz mono WAV..."
i=0
for f in "${files[@]}"; do
  i=$((i+1))
  # Hold out ~1 in 7 recordings as test data, the rest go to training
  if [ $((i % 7)) -eq 0 ]; then
    out="$POS_TEST/real_${i}.wav"
  else
    out="$POS_TRAIN/real_${i}.wav"
  fi
  ffmpeg -y -loglevel error -i "$f" -ar 16000 -ac 1 -sample_fmt s16 "$out"
done

n_train=$(find "$POS_TRAIN" -iname "real_*.wav" | wc -l)
n_test=$(find "$POS_TEST" -iname "real_*.wav" | wc -l)
echo ">>> Added $n_train real recordings to positive_train/, $n_test to positive_test/"
echo ">>> These get mixed in automatically the next time --augment_clips runs."
echo ">>> If features were already computed, re-run with --overwrite (see header above)."
