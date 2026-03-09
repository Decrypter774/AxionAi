# ⬡ Axion AI

An LLM built **entirely from scratch** in Python. Runs locally on a NVIDIA GPU. It has a tokenizer, transformer architecture, training loop, and a live chat web UI. 

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-orange?style=flat-square)
![CUDA](https://img.shields.io/badge/CUDA-12.4+-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-white?style=flat-square)

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/axion-ai.git
cd axion-ai
```

### 2. Create virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\Activate.ps1
# Mac/Linux:
source venv/bin/activate
```

### 3. Install PyTorch with CUDA
```bash
# For RTX 5070 / Blackwell (CUDA 12.8 nightly)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# For older GPUs (CUDA 12.4 stable)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
pip install flask flask-cors
```

---

## Training

### Quick test (no data needed)
```bash
python train.py --config small --batch_size 8 --epochs 5
```

### Train on TinyStories (recommended)
```bash
python -c "
from datasets import load_dataset
ds = load_dataset('roneneldan/TinyStories', split='train')
texts = [x['text'] for x in ds if len(x['text']) > 100][:50000]
open('tinystories.txt','w',encoding='utf-8').write('\n\n'.join(texts))
print(f'Saved {len(texts)} stories')
"

python train.py --config small --data tinystories.txt --vocab_size 4000 --batch_size 16 --grad_accum 2 --epochs 8 --lr 5e-4
```

### Resume training
```bash
python train.py --resume checkpoints/checkpoint_best.pt --data tinystories.txt --epochs 10 --lr 1e-4
```

### Model sizes

| Config | Params | VRAM |
|---|---|---|
| `small` | ~16M | ~4 GB |
| `medium` | ~130M | ~10 GB |
| `large` | ~350M | ~16 GB |

---

## Generate text

```bash
python generate.py --checkpoint checkpoints/checkpoint_best.pt --prompt "Once upon a time" --temperature 0.8
python generate.py --checkpoint checkpoints/checkpoint_best.pt --interactive
```

---

## Chat UI

```bash
python server.py --checkpoint checkpoints/checkpoint_best.pt
```

Open **http://localhost:5000**

---

## Requirements

- Python 3.10+
- NVIDIA GPU with 6GB+ VRAM
- CUDA 12.4+

---

## License

MIT — do whatever you want with it.

