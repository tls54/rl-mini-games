# Setup Instructions

This project uses Python 3.13 and requires PyTorch for device-agnostic training (MPS on Mac, CUDA on Linux/Colab, CPU fallback).

## Quick Start

### Mac / Linux (with pyenv)

```bash
# 1. Install Python 3.13.9 if needed
pyenv install 3.13.9

# 2. Navigate to project directory (pyenv will auto-select Python 3.13.9)
cd rl-mini-games

# 3. Create virtual environment
python -m venv .venv

# 4. Activate virtual environment
source .venv/bin/activate

# 5. Install dependencies
pip install -r requirements.txt
```

### Windows (without pyenv)

```bash
# 1. Download and install Python 3.13 from python.org

# 2. Navigate to project directory
cd rl-mini-games

# 3. Create virtual environment
python -m venv .venv

# 4. Activate virtual environment
.venv\Scripts\activate

# 5. Install dependencies
pip install -r requirements.txt
```

### Google Colab

```python
# In a Colab notebook cell:

# 1. Upgrade to Python 3.13 (if needed - Colab may require workarounds)
# Note: Colab typically runs 3.10, you may need to use 3.10 for Colab
# Alternatively, downgrade project to 3.10 for Colab compatibility

# 2. Clone your repo or upload files
!git clone https://github.com/your-username/rl-mini-games.git
%cd rl-mini-games

# 3. Install dependencies
!pip install -r requirements.txt

# 4. Verify device detection
import torch
print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
```

## Verifying Installation

```bash
# Check Python version
python --version  # Should show 3.13.x

# Check PyTorch and device
python -c "import torch; print(torch.__version__); print(torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'))"
```

Expected outputs:
- **Mac (M1/M2/M3)**: `mps` device
- **Linux/Windows with NVIDIA GPU**: `cuda` device
- **CPU fallback**: `cpu` device

## Development Dependencies (Optional)

```bash
# Install development tools
pip install pytest black ruff

# Or use pip install with extras
pip install -e ".[dev]"
```

## Troubleshooting

### Pyenv not switching to 3.13.9
```bash
# Check if .python-version is being read
cat .python-version

# Verify pyenv is in your PATH
which pyenv

# Manually set version
pyenv local 3.13.9
```

### PyTorch not detecting MPS on Mac
```bash
# Ensure you're on macOS 12.3+ with M1/M2/M3 chip
# Update PyTorch to latest version
pip install --upgrade torch
```

### Colab Python version mismatch
If Colab doesn't support Python 3.13:
1. Edit `pyproject.toml` and change `requires-python = ">=3.10"`
2. Use Colab's default Python 3.10
3. Test locally before pushing to ensure compatibility

## Project Structure
```
rl-mini-games/
├── envs/          # Game environments (tictactoe, connect4)
├── agents/        # RL agents (Q-learning, PPO)
├── utils/         # Device detection, helpers
├── train_*.py     # Training scripts
├── play.py        # Interactive play CLI
├── requirements.txt
├── pyproject.toml
└── .python-version
```
