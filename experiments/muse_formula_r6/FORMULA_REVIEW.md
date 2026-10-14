# R6 公式复核
准确恒等式：令g=sum g_k，e=y-g-mu，d=mu-c，则 r-S=e_t-bl*e_prev+(1-bl)d。若正确状态假设给E[e_t|past]=rho*a_prev，固定H=rho*a_prev-bl*e_prev+h0，理想h0=(1-bl)d；独立H截距校准实际偏移。GLS mu≠c，不默认为零。H_state/H_remainder两项加总为rho*a-bl*e；自由state额外H_calibration不必常数。
Sigma=v rho^lag+R I；平稳先验创新递推是精确Gaussian likelihood。在结构段OLS残差上拟合只是可行协方差估计，OLS估计误差未用REML校正，不能称受限极大似然。BIC在此作为训练前固定经验比较规则，不是正确模型选择保证。漏掉Gaussian AR协变量平方项时 Cov(beta(x²-Var x)_t, ..._(t-h))=2 beta² Var(x)² phi^(2h)，二阶相关并非自然AR源证明。
白化是对所有设计列（含截距）与y的同一线性算子W；W Sigma W'=I。一次GLS与初始OLS共享冻结组内SVD坐标；正定W保留零空间，不能修复漏基或秩亏。anchor clean总信号等价，因此共享协方差避免优化器数值误差制造规范差异，结构仍各自求解。
四点响应仅当前x,z变化，所有过去固定；真实残差和预测同一f0。保存原始源规范与训练中心化规范，有限生成源与过滤残差源分开；状态和观测余量贡献不是自然T或传感器误差。相关源可互相抵消，低预测MSE不保证内容恢复。
