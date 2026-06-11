# DiagnosticTS: Spatiotemporal Explicit Orthogonal Disentanglement for Multivariate Time-Series Forecasting

This repository contains the official implementation of the **DiagnosticTS** framework, designed for intrinsically interpretable multivariate time-series forecasting via spatiotemporal explicit orthogonal disentanglement and residual attribution.

## 🚀 Key Contributions
* **Lossless Disentanglement:** Achieves deep spatiotemporal coupling splitting with geometric orthogonal soft-constraints ($\mathcal{L}_{\text{ort}}$), driving latent alignment pureness ($Cos\_Sim^2 \to 0.002$) without sacrificing main forecasting accuracy.
* **Intrinsic Explainability:** Employs a dynamic gating network to isolate chronological pattern weights ($w_1$) from inter-series spatial impacts ($w_2$).

---

## 🛠️ Project Structure
```text
DiagnosticTS/
├── data_provider/       # Data loading and chronological sliding-window slicing
├── exp/                 # Operational center (Training and multi-model evaluation pipeline)
├── losses/              # Geometric orthogonal regularized constraints (info_losses.py)
├── models/              # Baseline and disentangled networks (proposed.py)
├── utils/               # Metric auditor (metrics.py) and complementary area-spectrum visualization
├── config.py            # Centralized hyperparameter definitions
└── main.py              # Main operations controller
⚡ Quick Start
1. Environment Setup
Bash
# Clone the repository
git clone [https://github.com/xzxmyr/Orthogonal-TSF.git](https://github.com/xzxmyr/Orthogonal-TSF.git)
cd DiagnosticTS

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch numpy pandas matplotlib
2. Run Pipeline
Place your target multi-variable sequential data (e.g., ETTm1.csv) under your designated path, modify config.py, and run the controller:

Bash
python main.py
📄 AI Transparency & Accountability Statement
In accordance with academic integrity frameworks, this repository utilizes Digital Intelligence (DI) tools for architectural logic modularization and visualization refactoring.

Human Intelligence (HI) Contribution: Mathematical proof formulation, spatiotemporal explicit encoder design, causal reasoning spectrum logic, and research conceptual framework are the independent work of the author.

Digital Intelligence (DI) Assistance: Code structure modularization, MLOps flattening refactoring, and Matplotlib backend visualization tuning guided by Gemini.
