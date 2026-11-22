#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_openwakeword.sh
#
# Sets up everything needed to train a custom openWakeWord model
# ("Hey Aaryan") locally, for use with Home Assistant.
#
# IMPORTANT: openWakeWord's synthetic-data pipeline (Piper TTS) only runs
# on Linux. On Windows, run this inside WSL2 (Ubuntu):
#     wsl --install -d Ubuntu
# then run this script from inside that Ubuntu shell.
#
# GPU (NVIDIA + CUDA) is optional but speeds training up a lot.
# CPU-only works fine for a single wake word, just slower (~30-60 min).
# ---------------------------------------------------------------------------
set -e

WORKDIR="$HOME/wakeword-training"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo ">>> Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip git ffmpeg libsndfile1 build-essential espeak-ng

echo ">>> Creating Python virtual environment..."
python3 -m venv oww-env
source oww-env/bin/activate

echo ">>> Cloning openWakeWord..."
if [ ! -d "openWakeWord" ]; then
  git clone https://github.com/dscripka/openWakeWord.git
fi
cd openWakeWord
pip install --upgrade pip

# NOTE: openWakeWord's setup.py hard-requires `speexdsp-ns` on Linux, but
# PyPI only ships wheels for a couple of specific Python versions/arches
# with no source dist to fall back to - `pip install -e .` fails on most
# systems with "No matching distribution found for speexdsp-ns". It's only
# used for optional noise suppression at inference time, not for training,
# so we install without deps and pull in everything else manually.
pip install -e . --no-deps
pip install 'onnxruntime>=1.10.0,<2' 'ai-edge-litert>=2.0.2,<3' \
            'tqdm>=4.0,<5.0' 'scipy>=1.3,<2' 'scikit-learn>=1,<2' \
            'requests>=2.0,<3'
pip install -r requirements.txt 2>/dev/null || true

echo ">>> Downloading core openWakeWord models (embedding + melspectrogram)..."
# NOTE: these don't ship in the git repo itself and aren't fetched
# automatically by train.py (only openwakeword.utils.download_models(),
# which train.py never calls, would normally grab them). Without this,
# --augment_clips fails with a "melspectrogram.onnx ... doesn't exist" error.
mkdir -p openwakeword/resources/models
( cd openwakeword/resources/models
  wget -nc https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx
  wget -nc https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite
  wget -nc https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx
  wget -nc https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite
)

echo ">>> Installing training-only requirements..."
# GPU build (NVIDIA, 8GB+ VRAM). If you're on WSL2, make sure you have the
# Windows NVIDIA driver installed (WSL2 uses it directly, no separate Linux
# driver needed) - see https://docs.nvidia.com/cuda/wsl-user-guide/index.html
#
# NOTE: torchaudio 2.9+ removed the classic `torchaudio.info()`/`.load()`
# top-level API in favor of torchcodec, but `torch_audiomentations` (used
# for the noise/RIR augmentation step) and openwakeword's own data.py still
# call the old API directly, breaking with:
#   AttributeError: module 'torchaudio' has no attribute 'info'
# Pinning to an older torchaudio isn't reliable here since torch/torchaudio
# wheels for brand-new Python versions (e.g. 3.14) often lag behind - so
# instead we leave torch/torchaudio unpinned and patch train.py below to
# shim .info()/.load() with soundfile-backed equivalents, which works
# regardless of torchaudio version.
pip install torch torchaudio
# Newer PyTorch's ONNX exporter (torch.onnx.export) now routes through the
# dynamo-based exporter, which depends on onnxscript - without it, exporting
# the final trained model fails with "ModuleNotFoundError: No module named
# 'onnxscript'" right at the very last step of training.
pip install onnxscript
pip install torchmetrics torchinfo speechbrain audiomentations torch-audiomentations \
            mutagen webrtcvad acoustics tqdm pyyaml scipy pronouncing deep-phonemizer \
            huggingface_hub datasets soundfile

# NOTE: the `acoustics` package imports `scipy.special.sph_harm` at module
# load time. That function was deprecated in scipy 1.15 and removed
# entirely in later versions, so an unpinned `pip install scipy` (which
# grabs the current latest) breaks `import acoustics` with:
#   ImportError: cannot import name 'sph_harm' from 'scipy.special'
# Pin to the last version that still has it, while staying compatible with
# the numpy>=2 requirement pulled in by piper-sample-generator.
pip install 'scipy==1.14.1'

echo ">>> Setting up a separate environment for ONNX -> TFLite conversion..."
# NOTE: train.py's built-in --convert_to_tflite step uses the old `onnx_tf` +
# `tensorflow-cpu==2.8.1`, which has no wheels for modern Python versions
# (e.g. 3.14) and is skipped entirely. Instead we use `onnx2tf`
# (github.com/PINTO0309/onnx2tf), an actively maintained converter - but it
# depends on TensorFlow, which also doesn't have 3.14 wheels yet, so this
# needs its own venv on an older Python (3.12).
#
# CRITICAL: run with plain `onnx2tf -i model.onnx -o out_dir` and the input
# tensor gets silently transposed (e.g. (1,16,96) -> (1,96,16)). Home
# Assistant's openWakeWord add-on (pyopen-wakeword) only checks buffer byte
# size when feeding the model, not axis order, so a transposed input model
# *loads and runs fine* but produces garbage predictions and never detects
# the wake word, at any threshold. The `-kat`/`--keep_shape_absolutely_input_names`
# flags (given the input tensor's actual name) prevent this.
sudo apt-get install -y python3.12 python3.12-venv
python3.12 -m venv "$WORKDIR/tflite-convert-env"
"$WORKDIR/tflite-convert-env/bin/pip" install --upgrade pip
"$WORKDIR/tflite-convert-env/bin/pip" install onnx2tf onnx onnxsim onnxruntime \
    'tensorflow==2.21.0' psutil ai-edge-litert

echo ">>> Verifying GPU is visible to PyTorch..."
python3 - <<'PY'
import torch
if torch.cuda.is_available():
    print(f"CUDA OK: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
else:
    print("WARNING: CUDA not detected. Training will fall back to CPU (much slower).")
    print("If you're on WSL2, check that the Windows NVIDIA driver is installed and")
    print("up to date, then re-run this script.")
PY

echo ">>> Installing piper-tts (synthetic TTS voice data)..."
# NOTE: openWakeWord's train.py was written against an older piper-sample-
# generator API that depended on a bundled `piper_train.vits` module (VITS
# training internals, not published on PyPI, needs a compiled monotonic_align
# extension). The current piper-sample-generator package (v3.x) *also*
# imports `piper_train` unconditionally at the top of its __main__.py, even
# though the ONNX-voice code path we want doesn't actually need it - so
# `from piper_sample_generator.__main__ import generate_samples_onnx` still
# fails with "No module named 'piper_train'" just from the import itself.
# To sidestep this entirely, we skip piper-sample-generator as a dependency
# and reimplement the (small) ONNX synthesis logic directly against
# `piper-tts`, which has no piper_train dependency at all.
pip install piper-tts

cd "$WORKDIR"
mkdir -p piper-sample-generator/models
cd piper-sample-generator

# Download several English Piper voices - mixing multiple voices/accents
# (not just multiple speakers within one voice) gives the trained wake word
# model more robustness to different real voices. NOTE: there is currently
# no Piper voice for Indian-accented English specifically (only en_US/en_GB
# exist) - the Hindi/Malayalam voices speak those languages, not English
# with an Indian accent, so they'd mispronounce "Hey Aaryan" rather than
# just sound accented. For genuine Indian-English coverage, use
# add_custom_recordings.sh with real recordings instead.
mkdir -p models
# en_US-libritts_r-medium: 904-speaker US English voice (primary, most diversity)
wget -nc -O models/en_US-libritts_r-medium.onnx \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx"
wget -nc -O models/en_US-libritts_r-medium.onnx.json \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx.json"
# en_GB-alba-medium: British English, female, single speaker
wget -nc -O models/en_GB-alba-medium.onnx \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx"
wget -nc -O models/en_GB-alba-medium.onnx.json \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx.json"
# en_GB-northern_english_male-medium: British English, male, single speaker
wget -nc -O models/en_GB-northern_english_male-medium.onnx \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx"
wget -nc -O models/en_GB-northern_english_male-medium.onnx.json \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx.json"

# Adapter so `from generate_samples import generate_samples` (used by
# openWakeWord's train.py) works, implemented directly against piper-tts's
# PiperVoice API (no piper_sample_generator/piper_train involved at all).
cat > generate_samples.py <<'PYEOF'
"""
Adapter exposing a `generate_samples(...)` function with the signature that
openWakeWord's train.py expects, implemented directly on top of piper-tts's
PiperVoice API (deliberately avoids piper_sample_generator's __main__.py,
which unconditionally imports the unpublished `piper_train` package even
when only the ONNX code path is used).

Cycles across MULTIPLE Piper voices/accents (not just multiple speakers
within one voice) for more variety - see _VOICE_FILES below.

Also resamples output to 16kHz: Piper voices synthesize at their own native
rate (22050 Hz for these particular voices), but openWakeWord's training
pipeline hard-requires every clip to be exactly 16000 Hz and raises a
ValueError otherwise.
"""
import itertools as it
import os
import wave
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as AF
from piper import PiperVoice, SynthesisConfig

_MODELS_DIR = Path(__file__).parent / "models"
_TARGET_SR = 16000

# Voices to mix together. LibriTTS-R has 904 speakers; later speaker IDs
# have very little training data and can produce artifacts, so cap it.
# NOTE: no Piper voice exists for Indian-accented English specifically -
# hi_IN/ml_IN voices speak Hindi/Malayalam, not English, and would
# mispronounce the phrase rather than just sound accented. For genuine
# Indian-English coverage, use add_custom_recordings.sh with real
# recordings instead.
_VOICE_FILES = [
    ("en_US-libritts_r-medium.onnx", 300),  # US English, 904 speakers (capped)
    ("en_GB-alba-medium.onnx", 1),           # British English, female
    ("en_GB-northern_english_male-medium.onnx", 1),  # British English, male
]

_voices = None  # list of (PiperVoice, num_speakers_to_use)


def _get_voices():
    global _voices
    if _voices is None:
        use_cuda = torch.cuda.is_available()
        _voices = []
        for filename, max_speakers in _VOICE_FILES:
            path = _MODELS_DIR / filename
            if not path.exists():
                continue  # skip voices that weren't downloaded
            voice = PiperVoice.load(str(path), use_cuda=use_cuda)
            num_speakers = min(voice.config.num_speakers or 1, max_speakers)
            _voices.append((voice, num_speakers))
        if not _voices:
            raise FileNotFoundError(f"No Piper voice models found in {_MODELS_DIR}")
    return _voices


def _synthesize_16k(voice, text, syn_config):
    """Synthesize `text` and return int16 PCM samples resampled to 16kHz."""
    chunks = [chunk.audio_float_array for chunk in voice.synthesize(text, syn_config=syn_config)]
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    native_sr = voice.config.sample_rate
    if native_sr != _TARGET_SR and audio.size > 0:
        t = torch.from_numpy(audio).unsqueeze(0)
        audio = AF.resample(t, orig_freq=native_sr, new_freq=_TARGET_SR).squeeze(0).numpy()
    return np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)


def generate_samples(
    text,
    max_samples,
    output_dir,
    file_names=None,
    batch_size=1,               # unused in the ONNX path, kept for API compatibility
    noise_scales=(0.667,),
    noise_scale_ws=(0.8,),
    length_scales=(0.75, 1.0, 1.25),
    auto_reduce_batch_size=False,  # unused in the ONNX path
    **kwargs,
):
    os.makedirs(output_dir, exist_ok=True)
    voices = _get_voices()

    if isinstance(text, str) and os.path.isfile(text):
        texts = it.cycle([line.strip() for line in open(text, encoding="utf-8") if line.strip()])
    elif isinstance(text, list):
        texts = it.cycle(text)
    else:
        texts = it.cycle([text])

    file_names_iter = it.cycle(file_names) if file_names else None
    settings_iter = it.cycle(it.product(length_scales, noise_scales, noise_scale_ws))

    # Round-robin through (voice, speaker_id) pairs across all loaded voices,
    # so samples are spread evenly across voices/accents rather than
    # exhausting one voice's speakers before moving to the next.
    voice_speaker_pairs = list(it.chain.from_iterable(
        [(voice, speaker_id) for speaker_id in range(num_speakers)]
        for voice, num_speakers in voices
    ))

    sample_idx = 0
    for voice, speaker_id in it.cycle(voice_speaker_pairs):
        if sample_idx >= max_samples:
            break
        length_scale, noise_scale, noise_w_scale = next(settings_iter)
        wav_path = Path(output_dir) / (next(file_names_iter) if file_names_iter else f"{sample_idx}.wav")

        syn_config = SynthesisConfig(
            speaker_id=speaker_id if voice.config.num_speakers > 1 else None,
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w_scale,
        )
        audio_int16 = _synthesize_16k(voice, next(texts), syn_config)

        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setframerate(_TARGET_SR)
            wav_file.setsampwidth(2)
            wav_file.setnchannels(1)
            wav_file.writeframes(audio_int16.tobytes())
        sample_idx += 1
PYEOF

echo ">>> Downloading Room Impulse Response (RIR) dataset (for reverb augmentation)..."
cd "$WORKDIR/openWakeWord"
pip install huggingface_hub datasets
pip install soundfile
python3 - <<'PY'
import datasets, os, soundfile as sf

# Note: newer `datasets` versions decode audio columns via torchcodec,
# returning an AudioDecoder object rather than a plain {'path','array'}
# dict. Also, in streaming mode the raw 'bytes' field is often None (only
# a reference is fetched lazily), so `decode(False)` doesn't reliably give
# us file bytes here. Instead, actually decode each clip to samples and
# write those out - this works consistently whether streamed or not, and
# we don't need to preserve the original filenames.
os.makedirs("mit_rirs", exist_ok=True)
rir_dataset = datasets.load_dataset("davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True)
for i, row in enumerate(rir_dataset):
    samples = row['audio'].get_all_samples()
    waveform = samples.data.cpu().numpy()  # shape: (channels, num_samples)
    sf.write(os.path.join("mit_rirs", f"rir_{i}.wav"), waveform.T, samples.sample_rate)
PY

echo ">>> Downloading background noise/music clips (for augmentation)..."
# Matches the official openWakeWord automatic_model_training notebook's data
# sources: a shard of AudioSet (general noise/speech/music) plus a slice
# of the Free Music Archive (music). Both get converted to 16kHz WAVs.
#
# Note: the agkphysics/AudioSet dataset repo no longer ships plain .tar
# shards (that layout is stale) - it's now stored as parquet files under
# data/bal_train/*.parquet. Loading directly via the generic "parquet"
# builder avoids depending on the repo's dataset loading script or its
# folder layout.
python3 - <<'PY'
import datasets, os, soundfile as sf

# Note: newer `datasets` decodes audio via torchcodec, returning an
# AudioDecoder object - use get_all_samples() to get actual waveform data
# rather than the old dict-style ['array'] access.

os.makedirs("audioset_16k", exist_ok=True)
audioset_dataset = datasets.load_dataset(
    "parquet",
    data_files="https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/bal_train/00.parquet",
    split="train",
    streaming=True,
)
audioset_dataset = audioset_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))

n_hours = 1  # bump this for a more robust model; 1 hour is a fast starting point
n_clips = n_hours * 3600 // 10  # AudioSet clips are ~10 seconds each
for i, row in enumerate(audioset_dataset):
    if i >= n_clips:
        break
    samples = row['audio'].get_all_samples()
    waveform = samples.data.cpu().numpy()
    sf.write(os.path.join("audioset_16k", f"audioset_{i}.wav"), waveform.T, samples.sample_rate)

# Note: the Free Music Archive mirror (rudraml/fma) relies on a legacy
# Python "dataset loading script", which current `datasets` versions no
# longer support at all ("Dataset scripts are no longer supported"). Using
# GTZAN (instrumental music, ~8 hours) and ESC-50 (environmental/ambient
# sound, ~2.8 hours) instead - both are stored in the modern parquet format.
os.makedirs("gtzan", exist_ok=True)
gtzan_dataset = datasets.load_dataset("sanchit-gandhi/gtzan", split="train", streaming=True)
gtzan_dataset = gtzan_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
for i, row in enumerate(gtzan_dataset):
    samples = row['audio'].get_all_samples()
    waveform = samples.data.cpu().numpy()
    sf.write(os.path.join("gtzan", f"gtzan_{i}.wav"), waveform.T, samples.sample_rate)

os.makedirs("esc50", exist_ok=True)
esc50_dataset = datasets.load_dataset("ashraq/esc50", split="train", streaming=True)
esc50_dataset = esc50_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
for i, row in enumerate(esc50_dataset):
    samples = row['audio'].get_all_samples()
    waveform = samples.data.cpu().numpy()
    sf.write(os.path.join("esc50", f"esc50_{i}.wav"), waveform.T, samples.sample_rate)
PY

echo ">>> Downloading precomputed negative feature set + false-positive validation set..."
# NOTE: the ACAV100M feature file is ~17GB. Using `wget -c` instead of
# hf_hub_download so an interrupted download (network drop, disk full,
# terminal closed) can simply be resumed by re-running this line, rather
# than silently leaving a missing/partial file that only surfaces later as
# a confusing FileNotFoundError deep inside train.py.
df -h . || true
wget -c -O openwakeword_features_ACAV100M_2000_hrs_16bit.npy \
  "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
wget -c -O validation_set_features.npy \
  "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy"

echo ">>> Patching train.py for PyTorch/torchaudio API compatibility..."
python3 - <<'PY'
path = "openwakeword/train.py"
src = open(path).read()

torch_load_patch = (
    "\nimport torch as _torch_patch\n"
    "_orig_torch_load = _torch_patch.load\n"
    "def _patched_torch_load(*args, **kwargs):\n"
    "    kwargs.setdefault('weights_only', False)\n"
    "    return _orig_torch_load(*args, **kwargs)\n"
    "_torch_patch.load = _patched_torch_load\n"
)
if "_patched_torch_load" not in src:
    src = src.replace("import torch\n", "import torch\n" + torch_load_patch, 1)

# Python 3.14 changed the default multiprocessing start method away from
# plain "fork" (which needs no pickling, since it clones process memory
# directly) to "forkserver" on some setups, which does require pickling.
# train.py's DataLoader workers get inline lambdas as label transforms that
# aren't picklable, causing:
#   _pickle.PicklingError: Can't pickle <function <lambda> ...>: it's not
#   found as __main__.<lambda>
# Forcing "fork" avoids this entirely.
mp_patch = (
    "\nimport multiprocessing as _mp_patch\n"
    "try:\n"
    "    _mp_patch.set_start_method('fork', force=True)\n"
    "except RuntimeError:\n"
    "    pass\n"
)
if "_mp_patch.set_start_method" not in src:
    src = src.replace("import torch\n", "import torch\n" + mp_patch, 1)

torchaudio_shim = '''
import torchaudio as _torchaudio_patch
import soundfile as _sf_patch
from collections import namedtuple as _namedtuple_patch

_AudioMetaData = _namedtuple_patch(
    "AudioMetaData", ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"]
)

def _shim_torchaudio_info(filepath, **kwargs):
    info = _sf_patch.info(str(filepath))
    return _AudioMetaData(
        sample_rate=info.samplerate, num_frames=info.frames,
        num_channels=info.channels, bits_per_sample=16, encoding="PCM_S",
    )

def _shim_torchaudio_load(filepath, frame_offset=0, num_frames=-1, **kwargs):
    data, sr = _sf_patch.read(
        str(filepath), start=frame_offset,
        frames=num_frames if num_frames and num_frames > 0 else -1,
        dtype="float32", always_2d=True,
    )
    tensor = _torch_patch.from_numpy(data.T).contiguous()
    return tensor, sr

# torchaudio 2.9+ removed the classic .info()/.load() API in favor of
# torchcodec, but torch_audiomentations and openwakeword.data still call
# them directly. Shim both with soundfile-backed equivalents so this works
# regardless of which torchaudio version got installed.
_torchaudio_patch.info = _shim_torchaudio_info
_torchaudio_patch.load = _shim_torchaudio_load
'''
if "_shim_torchaudio_load" not in src:
    marker = "_torch_patch.load = _patched_torch_load\n"
    src = src.replace(marker, marker + torchaudio_shim, 1)

open(path, "w").write(src)
print("train.py patched")
PY

echo ""
echo "=================================================================="
echo "Setup complete. Environment lives at: $WORKDIR/oww-env"
echo "Activate it in future sessions with:"
echo "    source $WORKDIR/oww-env/bin/activate"
echo ""
echo "Next: cd into $WORKDIR/openWakeWord and run the training command"
echo "using hey_aaryan.yml (see README.md and run_training.sh)."
echo "=================================================================="
