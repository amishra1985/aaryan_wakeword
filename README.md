# Training a custom "Hey Aaryan" wake word for Home Assistant

This uses [openWakeWord](https://github.com/dscripka/openWakeWord), the
open-source wake-word engine Home Assistant's Wyoming `openwakeword` add-on
uses. It trains entirely on synthetic speech (no need to record yourself
saying "Hey Aaryan" hundreds of times) and runs fully on your own machine.

## Requirements

- **Linux, or WSL2 on Windows.** The synthetic TTS step (Piper) doesn't run
  natively on Windows. Install with `wsl --install -d Ubuntu`, then do
  everything below inside that Ubuntu shell.
- Python 3.9+, ~10 GB free disk (for RIR/noise/feature datasets).
- You have an 8GB NVIDIA GPU, which `setup_openwakeword.sh` now installs
  CUDA-enabled PyTorch for — the whole pipeline (30,000 samples) should
  finish in well under an hour. If you're on WSL2, you only need the
  Windows NVIDIA driver installed; don't install a separate Linux driver.
  The script prints a check confirming your GPU is detected before training
  starts — if it says CUDA wasn't found, update your Windows NVIDIA driver
  and re-run.

## Files

- `setup_openwakeword.sh` — installs openWakeWord, piper-sample-generator,
  and downloads the shared datasets (room impulse responses, background
  noise, precomputed negative features, false-positive validation set).
- `hey_aaryan.yml` — training config: target phrase, negative phrases,
  dataset sizes, model architecture.
- `run_training.sh` — runs the 4-step pipeline (generate → augment →
  train → export).
- `add_custom_recordings.sh` — optional: mixes real recordings of you
  saying "Hey Aaryan" into the training set for better real-world accuracy
  (see "Improving real-world accuracy" below).

## Steps

```bash
# 1. Copy these three files into your Linux/WSL2 home directory, then:
bash setup_openwakeword.sh

# 2. Activate the environment and move into the cloned repo
source ~/wakeword-training/oww-env/bin/activate
cd ~/wakeword-training/openWakeWord

# 3. Put the config next to train.py and run the pipeline
cp /path/to/hey_aaryan.yml .
bash /path/to/run_training.sh
```

When it finishes you'll have:

```
my_custom_model/hey_aaryan.onnx
my_custom_model/hey_aaryan.tflite
```

## Tuning

- If the model triggers on unrelated speech (false positives), add the
  specific words/phrases that set it off to `custom_negative_phrases` in
  `hey_aaryan.yml` and re-run `run_training.sh` (or the individual
  `--generate_clips --augment_clips --train_model` steps, then the tflite
  conversion step).
- If it misses real activations (false negatives), raise `n_samples` to
  30,000-100,000 for more training variety.
- `target_false_positives_per_hour` and `max_negative_weight` control the
  precision/recall trade-off during the auto-training sequence.
- The add-on's `threshold` option (Settings → Add-ons → openWakeWord →
  Configuration) controls detection sensitivity at runtime, independent of
  training - lower it (e.g. 0.3) if a otherwise-working model needs to be
  more sensitive.

## Improving real-world accuracy with your own recordings

If the model only fires occasionally, or only at low confidence (checkable
via the add-on's debug logs), it's usually because it only ever saw
synthetic Piper TTS voices during training — not your actual voice, accent,
mic, or room acoustics. Mixing in a few dozen real recordings of yourself
saying "Hey Aaryan" is the single biggest lever for fixing this:

1. Record yourself saying "Hey Aaryan" ~30-50+ times in the room/mic you'll
   actually use it from (Windows Voice Recorder or a phone voice memo app
   both work). Keep each clip to just the phrase, ~1-2 seconds. If you want
   accent coverage beyond your own voice (e.g. Indian-accented English),
   this is the way to get it - Piper currently has no Indian-English voice
   to synthesize from (only US/British English exist; the Hindi/Malayalam
   voices speak those languages, so they'd mispronounce "Hey Aaryan" rather
   than just sound accented). Recording family/friends with similar accents
   is the reliable path here.
2. Copy them into one folder in your WSL2 home directory, e.g.
   `~/wakeword-training/my_recordings/` (reachable from Windows Explorer at
   `\\wsl$\Ubuntu\home\<you>\...`).
3. Run `bash add_custom_recordings.sh ~/wakeword-training/my_recordings` -
   it converts everything to 16kHz mono WAV and mixes it into the
   positive training/test data.
4. Re-run training (`bash run_training.sh`, or just the `--augment_clips
   --overwrite` and `--train_model` steps if clips were already augmented)
   and redeploy the new `.tflite` file.

Also worth trying independent of retraining: raise `n_samples` (30,000 →
60,000+) and `steps` (50,000 → 80,000+) in `hey_aaryan.yml` for more
synthetic variety on a longer run, and add background noise/audio that
matches your actual home environment to `background_paths`.

`setup_openwakeword.sh` also now downloads and mixes together three Piper
voices/accents (US English x2 genders across 300 speakers, plus British
English male and female) instead of just one, for broader baseline voice
coverage - see `_VOICE_FILES` in `generate_samples.py` if you want to add
more from the [Piper voice list](https://github.com/rhasspy/piper/blob/master/VOICES.md).

## Deploying to Home Assistant

Home Assistant uses openWakeWord via the **Wyoming Protocol** integration
(the `openwakeword` add-on if you're on Home Assistant OS/Supervised, or the
`wyoming-openwakeword` Docker container otherwise).

1. Copy `hey_aaryan.tflite` into the add-on's custom models folder - for the
   HAOS add-on this is the Samba `share` folder's `openwakeword`
   subdirectory (`\\<ha-ip>\share\openwakeword\`, i.e. `/share/openwakeword`
   on the host). **Must be `.tflite`** - the add-on only auto-loads that
   format from this folder; a `.onnx` file placed there is silently ignored.
2. Restart the openWakeWord add-on so it picks up the new model (Settings →
   Add-ons → openWakeWord → Restart). If the Wyoming integration doesn't
   reconnect afterward, see the troubleshooting note below.
3. In Home Assistant, go to **Settings → Voice Assistants**, edit your
   pipeline, and set **Wake word** to `hey_aaryan` (it will appear in the
   dropdown once the model is detected).
4. Each voice satellite/microphone device also has its own **Assistant**
   selector (a separate `select.*` entity, e.g.
   `select.assist_microphone_assistant`) that must be pointed at the same
   pipeline - the pipeline's "preferred" flag alone does not change which
   pipeline an existing satellite device uses.
5. Test by saying "Hey Aaryan" to that device.

## Notes / caveats

- The official Colab notebook version of this pipeline has occasionally
  broken between openWakeWord releases (see [issue #296](https://github.com/dscripka/openWakeWord/issues/296)).
  If `setup_openwakeword.sh` fails on a specific pip package version, check
  the [openWakeWord repo](https://github.com/dscripka/openWakeWord) for the
  current `requirements.txt` and adjust.
- `train.py` was written against an older `piper-sample-generator` API that
  depended on a bundled, never-published `piper_train.vits` module (needs a
  compiled `monotonic_align` extension). That path is broken on current
  setups, so the setup script instead installs `piper-tts` directly and adds
  a small adapter file (`generate_samples.py`) so `train.py` can call it
  unchanged (also handles resampling Piper's native 22050Hz output down to
  the 16000Hz training requires).
- **`train.py`'s own `--convert_to_tflite` is not used.** It depends on
  `onnx_tf` + `tensorflow-cpu==2.8.1`, which has no wheels for modern Python
  (e.g. 3.14). Instead, `setup_openwakeword.sh` sets up a separate Python
  3.12 venv (`tflite-convert-env`) with `onnx2tf`, an actively maintained
  alternative, and `run_training.sh` uses that for the final conversion step.
- **Critical onnx2tf gotcha:** running `onnx2tf` without extra flags
  silently *transposes* the model's input tensor (our classifier expects
  `(1, 16, 96)`; a plain conversion produces `(1, 96, 16)`). Home Assistant's
  openWakeWord add-on (built on `pyopen-wakeword`) only validates buffer
  *byte size* when feeding the model, not axis order - so a transposed model
  loads fine, streams audio fine, and simply never detects the wake word, at
  any threshold, with no error anywhere. `run_training.sh` passes
  `-kat <input_name> --keep_shape_absolutely_input_names <input_name>` to
  prevent this. If you ever convert a model manually and it "does nothing"
  despite loading correctly, check the `.tflite`'s input shape first:
  ```python
  import tensorflow as tf
  i = tf.lite.Interpreter(model_path="hey_aaryan.tflite")
  i.allocate_tensors()
  print(i.get_input_details()[0]['shape'])  # should be [1 16 96]
  ```
- Dataset filenames on the `davidscripka/openwakeword_features` Hugging
  Face dataset occasionally get renamed — if a download 404s, browse
  https://huggingface.co/datasets/davidscripka/openwakeword_features/tree/main
  and update the filename in `setup_openwakeword.sh`.
- If the Wyoming integration gets stuck showing "Failed setup, will retry:
  Unable to connect" after restarting the add-on, try, in order: restart the
  add-on again, reload the Wyoming config entry
  (`homeassistant.reload_config_entry` service), a full Home Assistant Core
  restart, or as a last resort delete the integration entry and let
  Supervisor's discovery recreate it (then redo step 3/4 above).

## Sources

- [dscripka/openWakeWord](https://github.com/dscripka/openWakeWord)
- [openWakeWord automatic_model_training.ipynb](https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb)
- [openWakeWord train.py](https://github.com/dscripka/openWakeWord/blob/main/openwakeword/train.py)
- [openWakeWord example custom_model.yml](https://github.com/dscripka/openWakeWord/blob/main/examples/custom_model.yml)
- [dscripka/piper-sample-generator](https://github.com/dscripka/piper-sample-generator)
