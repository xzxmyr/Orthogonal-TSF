# Surf / Orthogonal-TSF

冻结基础模型的残差建模与可识别性研究。**本次公开的最新已训练版本为 MUSE R6**：源码、完整模型参数、合成实验结果与失败证据均可复核。

R6 在初始 OLS 残差上拟合 Gaussian innovation likelihood，再冻结协方差进行一次 GLS；历史支路比较一阶残差、自由状态和固定状态公式。它是研究模型，不构成真实数据泛化、来源恢复、物理根因或部署证明。

- [最终总结与模型公式](FINAL_SUMMARY.md)
- [完整实验报告](experiments/muse_formula_r6/REPORT.md)
- [所有配置结果](experiments/muse_formula_r6/ALL_CONFIGURATIONS.md)
- [模型说明与复现步骤](experiments/muse_formula_r6/README.md)
- [完整模型下载：R6 2026-10-14](https://github.com/xzxmyr/Orthogonal-TSF/releases/tag/surf-r6-20261014)

## 直接使用已训练模型

从 Release 下载 `surf-r6-20261014.zip` 和对应的 `.sha256` 文件。ZIP 包含源码、全部 **2592 个复合模型、72 个冻结基础模型**及完整结果 JSON，不需要重新训练。

```bash
shasum -a 256 -c surf-r6-20261014.zip.sha256
unzip surf-r6-20261014.zip
cd surf-r6-20261014
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/muse_formula_r6/demo.py
python experiments/muse_formula_r6/verify_release.py
```

验证会重生固定合成数据，加载全部保存参数并核对预测指标、分量指标、账目、未来信息隔离与 SHA-256；不会拟合模型。NumPy/SciPy 版本锁定为实际核验环境，Python 使用 3.13。

Git 仓库保存源码、汇总和核验证据；完整 NPZ 参数及逐模型 `metrics.json` 放在 Release 包中。约 1.4 GB 的逐点预测缓存 `points.npz` 未分发，可按原 DGP 和参数重建。

## 核心结果与限制

R6 的 36 个有 AR、正确基配置中，平均预测 MSE：filtered state `0.24941384`，moment GLS state `0.24945451`，likelihood state `0.24943883`。似然相对矩 GLS 改善很小，filtered 对照仍更低。

强相关、无周期、有交互和 AR 的配置中，likelihood state 预测 MSE 为 `0.25021238`，但 M/J 源 MSE 达到 `2995.28688 / 514.11628`。重叠设计保持秩 `10/11`。这些失败与参数一并公开：预测准确不能推出分量归属准确。

本实验读取当前协变量及过去标签。实际预测必须确认当前协变量在预测时已经可得；这不是仅用历史输入的多步未来预报。

## 历史代码

`DiagnosticTS/` 保留原 GitHub 的早期 ETT 神经预测代码；它不是本次 R6 的依赖或当前推荐入口。早期 CMI/解耦表述未经当前模型验证，不应解释为独立性或因果证明。此版本未分发其外部数据与训练权重。

## License

第一方源码、文档及本次生成的合成模型参数采用 [MIT](LICENSE)。外部依赖受各自许可证约束，见 [NOTICE](NOTICE)。
