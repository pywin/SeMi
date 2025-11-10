# SeMi: When Imbalanced Semi-Supervised Learning Meets Mining Hard Examples

**SeMi** tackles **class-imbalanced semi-supervised learning (CISSL)** by combining **hard-example mining** with **imbalance-aware objectives**.  

---

## 📚 Table of Contents
- [⚙️ Environment](#environment)
- [🛠️ Installation](#installation)
- [🗄️ Datasets](#datasets)
- [🚀 Run Training](#-run-training)
- [Citation](#citation)
- [References](#references)
- [License](#license)

---

## ⚙️ Environment
Create and activate a clean conda environment (recommended):

```bash
# 1) Create a new environment
conda create -y -n semi python=3.8.20

# 2) Activate
conda activate semi

# 3) (Optional) Upgrade pip
python -m pip install -U pip wheel setuptools
```

---

## 🛠️ Installation
Install PyTorch (choose CUDA version as appropriate):

```bash
# Example for CUDA 12.x; visit https://pytorch.org/ for the right command
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

Install project dependencies:

```bash
# If you have a pinned requirements file:
pip install -r requirements.txt
```

---

## 🗄️ Datasets
Prepare dataset folders (examples below). You can store them anywhere and pass paths via configs.

```bash
# CIFAR
DATA_ROOT=./data
mkdir -p ${DATA_ROOT}
# CIFAR will be auto-downloaded by torchvision if not found.

# STL10 (auto-download)
# ImageNet-127 or LT variants require you to place them manually:
```

---

## 🚀 Run Training
Please copy the example commands below into your own script (e.g., `run_cifar10.sh` or `run_cifar100.sh`) according to the dataset and configuration you wish to train.

### CIFAR-10-LT
Run:
```bash
bash run_cifar10.sh
```

### CIFAR-100-LT
Run:
```bash
bash run_cifar100.sh
```

**Examples (CIFAR-10-LT, CIFAR-100-LT)**

**1) CIFAR10-LT (N=1500 M=3000 r=100):**
```bash
CUDA_VISIBLE_DEVICES=6 python train_semi.py --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 500 --val-iteration 500 --out ./results/cifar10/semi/N1500_r100/seed1_test --manualSeed 1 --decay_step 100 --warmup_step 2 --ws strong --alpha 0.4 --scale 0.75 --tau 0.7 --proto_t 0.09 --conswt 0.15 --num_samples 128 --use_pc 1 --df 0.99 --use_lcu 0 --use_lc 1 --use_mo 0
```

**2) CIFAR10-LT (N=500 M=4000 r=100):**
```bash
#CUDA_VISIBLE_DEVICES=1 python train_semi.py --ratio 8 --num_max 500 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 500 --val-iteration 500 --out ./results/cifar10/semi/N500_r100/seed1 --manualSeed 1 --decay_step 100 --warmup_step 20 --ws strong --alpha 0.4 --scale 0.75 --tau 0.7 --proto_t 0.09 --conswt 0.15 --num_samples 128 --use_pc 1 --df 0.99 --use_lcu 0 --use_lc 1 --use_mo 0
```

**3) CIFAR10-LT (N=150 M=300 r=10):**
```bash
CUDA_VISIBLE_DEVICES=0 python train_semi.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 10 --imb_ratio_u 10 --epoch 500 --val-iteration 500 --out ./results/cifar100/semi/N150_r10/seed1 --manualSeed 1 --decay_step 200 --warmup_step 20 --ws strong --alpha 0.4 --scale 0.75 --tau 0.7 --proto_t 0.09 --conswt 0.15 --num_samples 512 --use_pc 1 --df 0.99 --use_lcu 0 --use_lc 1 --use_mo 0
```

---

## Citation
If you find **SeMi** helpful, please consider citing:
```bibtex
@article{wang2025semi,
  title={SeMi: When Imbalanced Semi-Supervised Learning Meets Mining Hard Examples},
  author={Wang, Yin and Wang, Zixuan and Lu, Hao and Qin, Zhen and Zhao, Hailiang and Cheng, Guanjie and Su, Ge and Kuang, Li and Zhou, Mengchu and Deng, Shuiguang},
  journal={arXiv e-prints},
  pages={arXiv--2501},
  year={2025}
}
```

---

## References
https://github.com/YUE-FAN/CoSSL
https://github.com/google-research/fixmatch
https://github.com/ytaek-oh/daso

---

## License
This project is released under the **MIT License** (or your chosen license).  
See [LICENSE](LICENSE) for details.


