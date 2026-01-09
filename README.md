# PT-Herb

This project contains training code for Herb-LT dataset with long-tail recognition.

## Environment

### Requirements
- Python 3.9
- PyTorch 1.12.1+cu116
- torchvision 0.13.1+cu116
- torchaudio 0.12.1+cu116

### Installation

```bash
# Create conda environment
conda create -name ptenv python=3.9
conda activate ptenv

# Install PyTorch
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html

# Install dependencies
pip install timm==0.5.4
pip install opencv-python==4.6.0.66
pip install scikit-learn==1.0.2
pip install matplotlib==3.4.3
pip install pyyaml==6.0.1
pip install thop==0.1.1
```

## Usage

```bash
python main.py
```

## Project Structure

- `main.py` - Main training script
- `conv2Fpn.py` - Model architecture
- `loss/` - Loss functions
  - `contrastive.py` - Contrastive loss
  - `LabelSmoothing.py` - Label smoothing loss
