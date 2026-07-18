# DiagnosticTS: Spatiotemporal Explicit Orthogonal Disentanglement for Multivariate Time-Series Forecasting

---

## 🚀 Recent Updates & Breakthroughs (July 2026)
We have successfully upgraded the core codebase with an **Endogenous Residual Space Substitution** mechanism and an information-theoretic **Conditional Mutual Information (CMI)** bottleneck minimized via a variational **CLUB** estimator. 

### 📈 Empirical Performance Improvements (on ETTm1)
Compared with the baseline black-box architecture, our proposed framework achieves superior spatiotemporal purity and forecasting accuracy:

| Model Architecture | MSE ↓ | MAE ↓ | Estimated $I(z_1; z_2 \mid X)$ ↓ |
| :--- | :---: | :---: | :---: |
| **Baseline Black-Box** | 0.13017 | 0.24556 | — |
| **Proposed (DiagnosticTS)** | **0.10476** | **0.22054** | **3.71636** |

*   **Prediction Precision**: Achieved a **~19.5% relative reduction** in MSE.
*   **Decoupling Verification**: The conditional mutual information boundary successfully drops to **3.71636**, guaranteeing a near-orthogonal splitting of temporal and spatial latent spaces.
*   **Inherent Diagnostic Weights (Log-Entropy Space)**:
    *   *Temporal Periodic Inertia Share ($w_1$)*: **53.12%**
    *   *Spatial Cross-Variable Shock Share ($w_2$)*: **46.88%**

---

## ✨ Key Academic Contributions

1. **Lossless Disentanglement Constraint**: Implements a strict geometric orthogonal soft-constraint ($L_{ort}$) combined with a CMI upper-bound optimizer, driving latent alignment pureness without sacrificing forecasting accuracy.
2. **Intrinsic Explainability**: Employs a dynamic gating network to isolate chronological pattern weights ($w_1$) from cross-dimensional spatial topological correlation weights ($w_2$), providing authentic post-hoc attribution.

---

## 📂 Repository Structure
The repository is structured to maintain a clean, production-ready, and review-friendly layout:

```text
Orthogonal-TSF/
├── DiagnosticTS/           # Core source code directory
│   ├── data_provider/      # Data loading and preprocessing pipeline
│   ├── exp/                # Main training and evaluation loops
│   ├── losses/             # Loss functions (MSE, Orthogonal Constraint, CLUB CMI)
│   ├── models/             # Baseline & Proposed Network Architectures
│   ├── utils/              # Metrics, Visualization, and Math Tools
│   ├── config.py           # Hyperparameters and Configuration
│   └── main.py             # Main entry point for execution
├── data/                   # Place your local datasets here (e.g., ETTm1.csv)
└── README.md               # Project documentation

```

> **Note**: Local dataset files (e.g., `data/*.csv`) and PyTorch model checkpoints (`*.pth`) are untracked via `.gitignore` to keep the repository lightweight.

---

## 🛠️ Quick Start & Reproducibility

### 1. Environment Setup

Ensure you have Python 3.10+ and PyTorch installed. It is recommended to manage dependencies inside a virtual environment (`venv`).

### 2. Dataset Placement

Download the standard **ETTm1** dataset and place it inside the `data` directory:

```bash
mkdir -p data
# Place ETTm1.csv under the data/ directory

```

### 3. Training & Evaluation Pipeline

To reproduce the baseline results and execute our variational CMI audit module for the proposed architecture, simply run the following command from the root directory:

```bash
python DiagnosticTS/main.py

```

---

## 📑 Citation & Contact

If you find this repository or our spatiotemporal explicit anisotropy framework helpful for your research, please consider citing or reaching out.

```text
@article{xiangyu2026diagnosticts,
  title={Spatiotemporal Explicit Orthogonal Disentanglement for Multivariate Time-Series Forecasting},
  author={Xi, Xiangyu},
  journal={GitHub Repository: Orthogonal-TSF},
  year={2026}
}

```

```

```
