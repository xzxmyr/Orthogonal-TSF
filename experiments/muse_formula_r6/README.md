# MUSE R6 research model package

最新已训练模型；训练结果与限制见 [最终总结](../../FINAL_SUMMARY.md) 和 [REPORT](REPORT.md)。完整参数和逐模型指标从 [GitHub Release](https://github.com/xzxmyr/Orthogonal-TSF/releases/tag/surf-r6-20261014) 下载。

## 参数与接口

`artifacts/<configuration>/<variant>.npz` 是完整 `StateHistory` 模型。每个文件包含结构权重、训练期 SVD scale/rotation、冻结 base coef、协方差参数、mu、H 权重、规范与历史类型。加载采用 `allow_pickle=False`。

```python
from model import StateHistory
from run import dgp
import numpy as np

m = StateHistory.load("artifacts/s1429_regular_p1_j1_ar1/likelihood_clean_state.npz")
t, y, x, _ = dgp(1429, 1, 1, "regular", 1)
idx = np.arange(3601, 4801)
o = m.predict(t, y, x, idx)
f0 = m.coef[0] + m.coef[1] * t[idx] / 3600 + m.coef[2] * y[idx - 1]
y_hat = f0 + o["prediction_total"]
# Only after the current labels arrive:
unexplained = y[idx] - y_hat
```

从本目录运行以上示例，或从仓库根运行 `python experiments/muse_formula_r6/demo.py`。输出 `bias/T/M/J`、`structural_total`、`e_prev/a_prev`、`H_state/H_remainder/H_calibration/H` 和 `prediction_total`。

该接口按一步滚动评估，后续时点可以使用已到账的过去标签。当前协变量必须已经可得。合成真值仅用于验证和源误差评分，不是模型输入。所有候选均为研究对照，没有在保留段选择“部署最优模型”。

## 无训练核验

```bash
python experiments/muse_formula_r6/verify_release.py
```

完整 ZIP 必须先解压；仓库中不单独提交完整权重与 `metrics.json`。验证器从 DGP 再生数据，核对 72 配置、2592 模型库存、NPZ 与 base 参数一致、预测/结构/源 MSE、账目闭合、每模型一个单点的未来标签与协变量隔离、参数和文件不变，以及 `SHA256SUMS`。运行期间不写文件、不拟合。

`PUBLICATION_VERIFICATION.json` 是本次发布包的实际运行结果。`PUBLICATION_PROVENANCE.json` 区分原训练 checksum 与公开协议脱敏 checksum；`MODEL_INDEX.json` 列出全部 NPZ 哈希。

## 训练复现

原始 `run.py` 的单次执行保护会拒绝覆盖已经存在的 `artifacts/`。要复现实验，在全新的目录复制六个原始 Python 文件及协议、公式复核；该操作产生新的复现结果，不覆盖公开冻结参数。

```bash
fresh=$(mktemp -d)
cp experiments/muse_formula_r6/{model.py,run.py,preflight.py,audit_and_report.py,train_diagnostics.py,write_report.py,PROTOCOL_FROZEN.md,FORMULA_REVIEW.md} "$fresh/"
python "$fresh/preflight.py"
python "$fresh/run.py"
python "$fresh/audit_and_report.py"
python "$fresh/train_diagnostics.py"
python "$fresh/write_report.py"
```

全量复现会产生约 1.4 GB 的逐点缓存。原 `audit_and_report.py` 依赖这些缓存；精简 Release 应使用 `verify_release.py`。不同硬件或依赖环境会有浮点差异，报告应区分复现与新独立确认。
