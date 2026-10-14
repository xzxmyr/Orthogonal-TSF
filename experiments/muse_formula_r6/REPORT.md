# R6 创新残差似然与约束历史公式：实际训练报告
72新配置，种子1429/1543/1663，四段各1200。所有保留段结果现为开发信息；没有保留选模。旧R5不重训。

结构由当前时间基T、协变量一次/平方M、乘积J联合估计；f0=b0+bt*t/3600+bl*y_prev在base段冻结。生成层c+gT+gM+gJ，转换残差结构bias=c(1-bl)-b0、T=gT_cur-bl*gT_prev-bt*t/3600，M/J同过滤；filtered直接拟合该过滤残差。

新增协方差在结构段初始OLS残差上用精确Gaussian创新NLL，rho∈[.001,.975]、v/g0,R/g0∈[1e-6,4]；单起点最多160次评估，35迭代；IID与固定BIC门槛，数值/边界/预算失败回退IID。它是残差ML，不是REML或联合ML。冻结后一次GLS。锚点等价候选共享clean协方差，避免数值优化误差破坏规范等价。

输入t、当前/过去x,z、已观测过去y与冻结参数，无真实rho/A/eps。e_prev=y_prev-g_hat_prev-mu，mu=mean_structure(y-g_hat)，a_prev为处理e_prev后的posterior，当前prior=rho*a_prev。独立H段比较first=h0+h1e、自由state=h0+h1e+h2a、fixed=rho*a-bl*e+h0，仅拟合h0。真实改变了估计与函数约束，非坐标改名。

新增H_state=(rho-bl)*a、H_remainder=-bl*(e-a)、H_calibration=H-H_state-H_remainder。fixed校准项为常数；自由模型校准项可时变。输出bias/T/M/J、structural_total、e_prev/a_prev、以上H分列、H、prediction_total，以及f0+y_hat和标签后unexplained。状态模型贡献不是自然AR/传感器误差识别。

令d=mu-c，准确恒等式r-S=e_t-bl*e_prev+(1-bl)d。fixed理想常数为(1-bl)d，不能混用GLS c与mu。metrics逐候选保存mu、c、差异、理想h0与实际h0；filtered不虚构生成截距。

## 实际预算与核验
实际主训练计数：{'base': 72, 'generation_ols': 216, 'filtered_ols': 216, 'gls': 216, 'likelihood_gls': 216, 'likelihood_fit': 144, 'likelihood_nfev': 8464, 'H': 2592}。2592复合模型，各自保存加载；预检另有一个小样本优化调用，nfev=40。相同训练样本，额外计算不等于相同计算预算。
verification: passed=True。
independent_verification: passed=True。
likelihood_independent_verification: passed=True。
白化与密集Gaussian NLL、posterior索引、显式原始权重、H恒等式、单点批量、未来标签/协变量隔离、冻结映射、原始/中心化闭合、相同四点过去固定、锚点规范、组内冗余、跨组零空间支持内和离支持均核验。独立审计禁止fit调用。

## 全开clean同数据对照（三种子均值）
|设计|p|j|AR|候选|完整MSE|T源MSE|M源MSE|J源MSE|混合差分RMSE|
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
|regular|1|1|0|filtered_clean_first|0.061164|0.000042|0.000772|0.000151|0.001251|
|regular|1|1|0|filtered_clean_fixed|0.061098|0.000042|0.000772|0.000151|0.001251|
|regular|1|1|0|filtered_clean_state|0.061164|0.000042|0.000772|0.000151|0.001251|
|regular|1|1|0|generation_clean_first|0.061188|0.000074|0.000173|0.000011|0.000297|
|regular|1|1|0|generation_clean_fixed|0.061112|0.000074|0.000173|0.000011|0.000297|
|regular|1|1|0|generation_clean_state|0.061188|0.000074|0.000173|0.000011|0.000297|
|regular|1|1|0|gls_clean_first|0.061188|0.000074|0.000173|0.000011|0.000297|
|regular|1|1|0|gls_clean_fixed|0.061112|0.000074|0.000173|0.000011|0.000297|
|regular|1|1|0|gls_clean_state|0.061188|0.000074|0.000173|0.000011|0.000297|
|regular|1|1|0|likelihood_clean_first|0.061188|0.000074|0.000173|0.000011|0.000297|
|regular|1|1|0|likelihood_clean_fixed|0.061112|0.000074|0.000173|0.000011|0.000297|
|regular|1|1|0|likelihood_clean_state|0.061188|0.000074|0.000173|0.000011|0.000297|
|regular|1|1|1|filtered_clean_first|0.253547|0.012123|0.001413|0.000169|0.001055|
|regular|1|1|1|filtered_clean_fixed|0.249835|0.012123|0.001413|0.000169|0.001055|
|regular|1|1|1|filtered_clean_state|0.249618|0.012123|0.001413|0.000169|0.001055|
|regular|1|1|1|generation_clean_first|0.254616|0.013405|0.003100|0.000355|0.001667|
|regular|1|1|1|generation_clean_fixed|0.250976|0.013405|0.003100|0.000355|0.001667|
|regular|1|1|1|generation_clean_state|0.250654|0.013405|0.003100|0.000355|0.001667|
|regular|1|1|1|gls_clean_first|0.253639|0.013135|0.001589|0.000062|0.000800|
|regular|1|1|1|gls_clean_fixed|0.250005|0.013135|0.001589|0.000062|0.000800|
|regular|1|1|1|gls_clean_state|0.249634|0.013135|0.001589|0.000062|0.000800|
|regular|1|1|1|likelihood_clean_first|0.253642|0.013139|0.001594|0.000064|0.000815|
|regular|1|1|1|likelihood_clean_fixed|0.249913|0.013139|0.001594|0.000064|0.000815|
|regular|1|1|1|likelihood_clean_state|0.249619|0.013139|0.001594|0.000064|0.000815|
|correlated|1|1|0|filtered_clean_first|0.061418|0.001608|0.019676|0.010085|0.012977|
|correlated|1|1|0|filtered_clean_fixed|0.061362|0.001608|0.019676|0.010085|0.012977|
|correlated|1|1|0|filtered_clean_state|0.061418|0.001608|0.019676|0.010085|0.012977|
|correlated|1|1|0|generation_clean_first|0.061013|0.000971|0.011801|0.000786|0.003651|
|correlated|1|1|0|generation_clean_fixed|0.060940|0.000971|0.011801|0.000786|0.003651|
|correlated|1|1|0|generation_clean_state|0.061013|0.000971|0.011801|0.000786|0.003651|
|correlated|1|1|0|gls_clean_first|0.061013|0.000971|0.011801|0.000786|0.003651|
|correlated|1|1|0|gls_clean_fixed|0.060940|0.000971|0.011801|0.000786|0.003651|
|correlated|1|1|0|gls_clean_state|0.061013|0.000971|0.011801|0.000786|0.003651|
|correlated|1|1|0|likelihood_clean_first|0.061013|0.000971|0.011801|0.000786|0.003651|
|correlated|1|1|0|likelihood_clean_fixed|0.060940|0.000971|0.011801|0.000786|0.003651|
|correlated|1|1|0|likelihood_clean_state|0.061013|0.000971|0.011801|0.000786|0.003651|
|correlated|1|1|1|filtered_clean_first|0.253199|0.037578|0.139346|0.042982|0.023328|
|correlated|1|1|1|filtered_clean_fixed|0.249625|0.037578|0.139346|0.042982|0.023328|
|correlated|1|1|1|filtered_clean_state|0.249229|0.037578|0.139346|0.042982|0.023328|
|correlated|1|1|1|generation_clean_first|0.256467|0.045636|0.309946|0.048408|0.023146|
|correlated|1|1|1|generation_clean_fixed|0.252869|0.045636|0.309946|0.048408|0.023146|
|correlated|1|1|1|generation_clean_state|0.252441|0.045636|0.309946|0.048408|0.023146|
|correlated|1|1|1|gls_clean_first|0.253354|0.037270|0.113488|0.041809|0.020350|
|correlated|1|1|1|gls_clean_fixed|0.249868|0.037270|0.113488|0.041809|0.020350|
|correlated|1|1|1|gls_clean_state|0.249330|0.037270|0.113488|0.041809|0.020350|
|correlated|1|1|1|likelihood_clean_first|0.253352|0.037231|0.111360|0.041453|0.020340|
|correlated|1|1|1|likelihood_clean_fixed|0.249775|0.037231|0.111360|0.041453|0.020340|
|correlated|1|1|1|likelihood_clean_state|0.249313|0.037231|0.111360|0.041453|0.020340|
|overlap|1|1|0|filtered_clean_first|0.061848|0.000048|0.000295|0.000592|0.002303|
|overlap|1|1|0|filtered_clean_fixed|0.061789|0.000048|0.000295|0.000592|0.002303|
|overlap|1|1|0|filtered_clean_state|0.061848|0.000048|0.000295|0.000592|0.002303|
|overlap|1|1|0|generation_clean_first|0.061132|0.033793|0.058287|0.000109|0.001214|
|overlap|1|1|0|generation_clean_fixed|0.061059|0.033793|0.058287|0.000109|0.001214|
|overlap|1|1|0|generation_clean_state|0.061132|0.033793|0.058287|0.000109|0.001214|
|overlap|1|1|0|gls_clean_first|0.061132|0.033793|0.058287|0.000109|0.001214|
|overlap|1|1|0|gls_clean_fixed|0.061059|0.033793|0.058287|0.000109|0.001214|
|overlap|1|1|0|gls_clean_state|0.061132|0.033793|0.058287|0.000109|0.001214|
|overlap|1|1|0|likelihood_clean_first|0.061132|0.033793|0.058287|0.000109|0.001214|
|overlap|1|1|0|likelihood_clean_fixed|0.061059|0.033793|0.058287|0.000109|0.001214|
|overlap|1|1|0|likelihood_clean_state|0.061132|0.033793|0.058287|0.000109|0.001214|
|overlap|1|1|1|filtered_clean_first|0.252275|0.001150|0.001453|0.000244|0.001631|
|overlap|1|1|1|filtered_clean_fixed|0.248636|0.001150|0.001453|0.000244|0.001631|
|overlap|1|1|1|filtered_clean_state|0.248364|0.001150|0.001453|0.000244|0.001631|
|overlap|1|1|1|generation_clean_first|0.253412|0.043453|0.096902|0.014452|0.011864|
|overlap|1|1|1|generation_clean_fixed|0.249673|0.043453|0.096902|0.014452|0.011864|
|overlap|1|1|1|generation_clean_state|0.249194|0.043453|0.096902|0.014452|0.011864|
|overlap|1|1|1|gls_clean_first|0.252529|0.043164|0.083152|0.000637|0.002348|
|overlap|1|1|1|gls_clean_fixed|0.248972|0.043164|0.083152|0.000637|0.002348|
|overlap|1|1|1|gls_clean_state|0.248462|0.043164|0.083152|0.000637|0.002348|
|overlap|1|1|1|likelihood_clean_first|0.252531|0.043162|0.083153|0.000621|0.002320|
|overlap|1|1|1|likelihood_clean_fixed|0.248866|0.043162|0.083153|0.000621|0.002320|
|overlap|1|1|1|likelihood_clean_state|0.248447|0.043162|0.083153|0.000621|0.002320|

## 所有开关clean整体对照
|设计|AR|估计器|state MSE|fixed MSE|fixed胜state/12|
|---|---:|---|---:|---:|---:|
|regular|0|generation|0.061196|0.061126|4/12|
|regular|0|filtered|0.061263|0.061199|6/12|
|regular|0|gls|0.061196|0.061126|4/12|
|regular|0|likelihood|0.061196|0.061126|4/12|
|regular|1|generation|0.251692|0.251991|8/12|
|regular|1|filtered|0.249949|0.250153|8/12|
|regular|1|gls|0.249932|0.250316|8/12|
|regular|1|likelihood|0.249915|0.250226|8/12|
|correlated|0|generation|0.061109|0.061040|4/12|
|correlated|0|filtered|0.061615|0.061575|5/12|
|correlated|0|gls|0.061109|0.061040|4/12|
|correlated|0|likelihood|0.061109|0.061040|4/12|
|correlated|1|generation|0.252585|0.252937|8/12|
|correlated|1|filtered|0.249717|0.250011|8/12|
|correlated|1|gls|0.249780|0.250248|8/12|
|correlated|1|likelihood|0.249763|0.250157|8/12|
|overlap|0|generation|0.061146|0.061080|4/12|
|overlap|0|filtered|0.061311|0.061248|6/12|
|overlap|0|gls|0.061146|0.061080|4/12|
|overlap|0|likelihood|0.061146|0.061080|4/12|
|overlap|1|generation|0.249953|0.250362|8/12|
|overlap|1|filtered|0.248575|0.248811|7/12|
|overlap|1|gls|0.248652|0.249117|8/12|
|overlap|1|likelihood|0.248638|0.249017|8/12|

## 协方差选择与失败（每规范36配置）
|规范|AR|方法|接受AR|回退原因|实际nfev|
|---|---:|---|---:|---|---:|
|clean|0|gls|0|{'weak_lag1': 36}|0|
|clean|0|likelihood|0|{'boundary_solution': 36}|3612|
|clean|1|gls|36|{}|0|
|clean|1|likelihood|36|{}|1408|
|underfit|0|gls|26|{'weak_lag1': 6, 'negative_variance': 2, 'unstable_or_nonpositive_rho': 2}|0|
|underfit|0|likelihood|28|{'boundary_solution': 8}|2012|
|underfit|1|gls|36|{}|0|
|underfit|1|likelihood|36|{}|1432|
|anchor|0|gls|0|{'weak_lag1': 36}|0|
|anchor|0|likelihood|0|{'boundary_solution': 36}|0|
|anchor|1|gls|36|{}|0|
|anchor|1|likelihood|36|{}|0|

无真实AR的漏M平方残差可以有真实序列相关，不一概称纯噪声误报。Gaussian AR协变量遗漏平方项协方差正比phi^(2h)。创新似然选AR不表明来源正确。

## 强相关无周期有乘积有AR：必须同时看源失败
|设计|p|j|AR|候选|完整MSE|T源MSE|M源MSE|J源MSE|混合差分RMSE|
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
|correlated|0|1|1|filtered_anchor_state|0.250315|0.112054|525.120014|300.735667|3.560691|
|correlated|0|1|1|filtered_clean_state|0.250315|0.112054|3235.032742|560.198057|3.560691|
|correlated|0|1|1|filtered_underfit_state|0.249472|0.113033|6.034853|0.561978|0.113941|
|correlated|0|1|1|generation_anchor_state|0.252730|0.113972|1427.135261|807.081138|5.823218|
|correlated|0|1|1|generation_clean_state|0.252730|0.113972|8955.785014|1540.300314|5.823218|
|correlated|0|1|1|generation_underfit_state|0.250426|0.117102|8.533796|0.577059|0.115562|
|correlated|0|1|1|gls_anchor_state|0.250229|0.113423|488.414580|276.944752|3.367735|
|correlated|0|1|1|gls_clean_state|0.250229|0.113423|2984.316935|512.000883|3.367735|
|correlated|0|1|1|gls_underfit_state|0.249482|0.114077|5.397120|0.559649|0.113679|
|correlated|0|1|1|likelihood_anchor_state|0.250212|0.113452|490.289411|278.068620|3.376436|
|correlated|0|1|1|likelihood_clean_state|0.250212|0.113452|2995.286878|514.116278|3.376436|
|correlated|0|1|1|likelihood_underfit_state|0.249465|0.114074|5.394214|0.559651|0.113679|

## 原始与训练中心化生成源：likelihood clean state全开
|设计|AR|T raw/centered|M raw/centered|J raw/centered|
|---|---:|---|---|---|
|regular|0|0.003424/0.001306|0.000369/0.000213|0.000028/0.000017|
|regular|1|0.608932/0.208476|0.005062/0.001790|0.000170/0.000092|
|correlated|0|0.018124/0.016393|0.229678/0.012754|0.002038/0.001196|
|correlated|1|0.734257/0.320740|1.423269/0.106415|0.090449/0.059088|
|overlap|0|2.902958/2.900689|26.279253/2.879939|0.000351/0.000173|
|overlap|1|4.042953/3.638394|27.039150/3.256577|0.001911/0.001051|

## 状态与代理诊断：likelihood clean state全开
|设计|AR|代理污染MSE|state对真AR MSE|H对潜在oracle MSE|mu-c均值|
|---|---:|---:|---:|---:|---:|
|regular|0|0.001507|0.000000|0.000654|0.000000|
|regular|1|0.210776|0.270072|0.030006|0.001501|
|correlated|0|0.001239|0.000000|0.000453|0.000000|
|correlated|1|0.217460|0.277581|0.032741|0.001406|
|overlap|0|0.001606|0.000000|0.000950|0.000000|
|overlap|1|0.212491|0.272052|0.031084|0.001295|

代理真值为过去AR+noise减结构段真误差均值；state对未中心化真AR诊断含规范偏移；H潜在oracle使用不可观测真实过去AR/noise，不能当同观测信息集可达基准。metrics保存所有规范/开关/候选的源原始、训练中心化MSE/bias/RMS、生成层、代理、state/H、完整预测及固定历史四点。ALL_CONFIGURATIONS.md含全部864个三种子汇总；不能仅看主表。

## 训练稳定性与结论边界
train_stability.json为864结构的训练段设计诊断；conditional_train_source_variance使用已拟合协方差，不含其估计不确定性或错模型风险。秩亏协方差仅为所选伪逆规范，null敏感性更关键；不用于选惩罚或候选。近共线、漏基与组间零空间仍未修复。低预测MSE可与巨大源误差和响应失真同时存在；创新似然/H约束不自动解决内容归属。

三固定种子合成一步滚动，没有总体、显著性、因果、真实数据、物理测量误差或部署证明。本报告记录 R6 实验快照；后续研究仍待验证。
