#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_training.sh
#
# Runs the full openWakeWord training pipeline for "Hey Aaryan" using the
# config in hey_aaryan.yml. Run this from inside $WORKDIR/openWakeWord
# (after setup_openwakeword.sh has completed and the venv is active):
#
#   source ~/wakeword-training/oww-env/bin/activate
#   cd ~/wakeword-training/openWakeWord
#   cp /path/to/hey_aaryan.yml .
#   bash /path/to/run_training.sh
#
# Steps run in order (each can also be run individually via train.py flags):
#   1. --generate_clips : synthesize positive + adversarial-negative speech
#   2. --augment_clips  : add noise/reverb, convert to openWakeWord features
#   3. --train_model    : train the classifier + auto-select best checkpoint,
#                          exports hey_aaryan.onnx
#   4. onnx2tf          : convert to .tflite in the separate tflite-convert-env
#                          (train.py's own --convert_to_tflite is not used -
#                          see setup_openwakeword.sh for why)
#
# TIP: if detection is unreliable/low-confidence on your actual voice, run
#   add_custom_recordings.sh first to mix in real recordings of yourself
#   saying "Hey Aaryan" - synthetic TTS clips alone often aren't enough.
#   See README.md, "Improving real-world accuracy with your own recordings".
# ---------------------------------------------------------------------------
set -e

CONFIG="hey_aaryan.yml"
WORKDIR="$HOME/wakeword-training"
MODEL_NAME="hey_aaryan"
MODEL_DIR="my_custom_model"

echo ">>> Step 1/4: generating synthetic training clips (this takes a while)..."
# python3 openwakeword/train.py --training_config "$CONFIG" --generate_clips

echo ">>> Step 2/4: augmenting clips + computing features..."
# python3 openwakeword/train.py --training_config "$CONFIG" --augment_clips

echo ">>> Step 3/4: training model (exports .onnx)..."
# python3 openwakeword/train.py --training_config "$CONFIG" --train_model

echo ">>> Step 4/4: converting .onnx to .tflite (in a separate Python 3.12 env)..."
if [ -x "$WORKDIR/tflite-convert-env/bin/onnx2tf" ]; then
  ONNX_PATH="$MODEL_DIR/$MODEL_NAME.onnx"

  # Detect the input tensor's actual name rather than assuming "x", since
  # this can vary between exports - needed for the -kat / axis-preserving
  # flags below.
  INPUT_NAME=$("$WORKDIR/tflite-convert-env/bin/python3" - <<PY
import onnx
m = onnx.load("$ONNX_PATH")
print(m.graph.input[0].name)
PY
)
  echo ">>> Detected input tensor name: $INPUT_NAME"

  # CRITICAL: without -kat/--keep_shape_absolutely_input_names, onnx2tf
  # silently transposes the input (e.g. (1,16,96) -> (1,96,16)). The model
  # still loads and runs in Home Assistant's openWakeWord add-on (it only
  # checks buffer byte size, not axis order) but produces garbage
  # predictions and never detects the wake word at any threshold. Ask us
  # how we know.
  "$WORKDIR/tflite-convert-env/bin/onnx2tf" \
    -i "$ONNX_PATH" \
    -o "$MODEL_DIR/${MODEL_NAME}_tflite_out" \
    -kat "$INPUT_NAME" \
    --keep_shape_absolutely_input_names "$INPUT_NAME"

  cp "$MODEL_DIR/${MODEL_NAME}_tflite_out/${MODEL_NAME}_float32.tflite" \
     "$MODEL_DIR/$MODEL_NAME.tflite"

  # Sanity-check the resulting shape before declaring success
  "$WORKDIR/tflite-convert-env/bin/python3" - <<PY
import tensorflow as tf
i = tf.lite.Interpreter(model_path="$MODEL_DIR/$MODEL_NAME.tflite")
i.allocate_tensors()
shape = i.get_input_details()[0]['shape']
print(f"Converted .tflite input shape: {shape}")
if list(shape) != [1, 16, 96]:
    print("WARNING: expected input shape [1, 16, 96] - got something else.")
    print("The -kat fix may not have applied correctly; do not deploy this file as-is.")
PY

  TFLITE_AVAILABLE=1
else
  echo ">>> tflite-convert-env not found (see setup_openwakeword.sh) - skipping .tflite export."
  echo ">>> Home Assistant's openWakeWord add-on requires a .tflite file, so you'll need to"
  echo ">>> run the conversion manually later - see README.md."
  TFLITE_AVAILABLE=0
fi

echo ""
echo "=================================================================="
echo "Done. Your trained model file(s) are in: ./$MODEL_DIR/"
echo "  - $MODEL_NAME.onnx"
[ "$TFLITE_AVAILABLE" = "1" ] && echo "  - $MODEL_NAME.tflite  <- copy this one into Home Assistant"
echo ""
echo "Home Assistant's openWakeWord add-on only auto-loads *.tflite* files"
echo "from /share/openwakeword (not .onnx). See README.md for deployment steps."
echo "=================================================================="
