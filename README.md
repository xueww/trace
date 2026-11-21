# TRACE-DTI

Official implementation and prospective evaluation pipeline for **TRACE-DTI: Topological Relational Distillation and Linear Cross-Attention for High-Throughput Drug–Target Interaction Prediction**.

This repository contains the inference architecture, configuration files, and evaluation scripts accompanying the study.

---

## ⚙️ Environment Setup

We recommend setting up an isolated virtual environment using Conda (Python 3.9 or 3.10):

```bash
# 1. Create and activate a conda environment
conda create -n tracedti python=3.10 -y
conda activate tracedti

# 2. Install PyTorch with CUDA support (adjust according to your CUDA driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install remaining dependencies
pip install -r requirements_inference.txt
```

---

## 📦 Data and Pre-trained Checkpoints

Due to file size constraints, pre-trained model weights and prepared test evaluation payloads (`.pt`) are hosted on Google Drive:

* **Google Drive Link**: [Download Datasets and Checkpoints](https://drive.google.com/drive/folders/1KMm71SPq7_2tat-RqkQf4R3Zzad9HiJI?usp=sharing)

After downloading, please extract the archive into the `./checkpoint/` directory.

---

## 🚀 Quick Start & Metric Reproduction

The `main.py` script evaluates trained TRACE-DTI checkpoints on the prepared test datasets and reproduces the performance metrics reported in the paper.

### 1. Evaluate S1 Random Interaction-Level Split
Evaluate all 5 folds sequentially:
```bash
python main.py --dataset Kinase --scenario cv1 --fold 0 --device cuda:0
```
Evaluate a specific fold (e.g., Fold 1):
```bash
python main.py --dataset Kinase --scenario cv1 --fold 1 --device cuda:0
```

### 2. Evaluate Cold-Start Scenarios
* **S2 Cold Target (Unseen Protein)**:
  ```bash
  python main.py --dataset Kinase --scenario s2 --fold 1 --device cuda:0
  ```
* **S3 Cold Drug (Unseen Compound)**:
  ```bash
  python main.py --dataset Kinase --scenario s3 --fold 1 --device cuda:0
  ```
* **S4 Dual Cold (Unseen Compound & Unseen Protein)**:
  ```bash
  python main.py --dataset Kinase --scenario s4 --fold 1 --device cuda:0
  ```

---

## 🔧 Command-Line Arguments

| Argument | Default | Description |
| :--- | :---: | :--- |
| `--dataset` | `Kinase` | Benchmark dataset (`Kinase`, `biosnap`, or `Bank`). |
| `--scenario` | `cv1` | Evaluation protocol: `cv1` (S1 Random), `s2` (Cold Target), `s3` (Cold Drug), `s4` (Dual Cold). |
| `--fold` | `0` | Fold index to evaluate (`1-5` for an individual fold; `0` runs all 5 folds). |
| `--device` | `cuda:0` | Computing device (`cuda:0` or `cpu`). |
| `--batch-size` | `128` | Inference batch size. |
| `--output-dir` | `None` | Custom directory containing test payloads (defaults to `checkpoint/<scenario>`). |
| `--checkpoint-dir` | `None` | Custom directory containing checkpoint `.pt` files. |

---

## 📊 Output Results

During execution, `main.py` calculates standard classification and ranking metrics:
* **Metrics**: AUROC, AUPRC, Accuracy, Precision, Recall, F1-score, Matthews Correlation Coefficient (MCC), and the optimal decision threshold.
* **Per-fold Output**: `info_test{fold}.csv` containing ground-truth labels, prediction scores, and binary predictions.
* **Summary Table**: `info_metrics.csv` containing overall cross-validation metrics across all evaluated folds.

---

## 🔒 Confidentiality & Reproducibility Notice

This repository is provided strictly for academic research and peer-review evaluation under the CC BY-NC 4.0 license (non-commercial use only). Commercial exploitation or unauthorized redistribution is strictly prohibited. Full open-source release under the CC BY-NC 4.0 License will accompany formal publication.
