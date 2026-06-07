#!/usr/bin/env bash
# Sets up the training virtual environment on the GPU host GPU nodes.
# Run once before training: bash setup_train_env.sh
set -e

BASE="/awshesh/lfm2.5/awshesh"
cd "$BASE"

echo "========================================"
echo " LFM2.5 Training Environment Setup"
echo "========================================"

source .venv_train/bin/activate

echo "[1/3] Installing PyTorch + torchaudio (CUDA 12.1)..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "[2/3] Installing training dependencies..."
uv pip install \
    liquid-audio \
    peft \
    transformers \
    accelerate \
    soundfile \
    librosa \
    wandb \
    numpy \
    tqdm \
    huggingface_hub \
    jiwer

echo "[3/3] Verifying install..."
python - <<'EOF'
import torch, peft, transformers, liquid_audio, soundfile, wandb
print(f"  torch         : {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU           : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM (GB)     : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}")
print(f"  peft          : {peft.__version__}")
print(f"  transformers  : {transformers.__version__}")
print(f"  liquid_audio  : {getattr(liquid_audio,'__version__','ok')}")
EOF

echo ""
echo "Done. Activate with: source .venv_train/bin/activate"
