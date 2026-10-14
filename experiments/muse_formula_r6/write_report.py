from pathlib import Path
import json,statistics as st,collections
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'artifacts'
rows=json.loads((OUT/'metrics.json').read_text());agg=json.loads((OUT/'aggregate_metrics.json').read_text());line=json.loads((OUT/'execution_lineage.json').read_text());lookup={(r['id'],r['variant']):r for r in rows}
def mean(rr,key):return st.mean(key(r) for r in rr)
def table(qq):
 text='|设计|p|j|AR|候选|完整MSE|T源MSE|M源MSE|J源MSE|混合差分RMSE|\n|---|---:|---:|---:|---|---:|---:|---:|---:|---:|\n'
 for q in qq:
  text+=f"|{q['design']}|{q['period']}|{q['interaction']}|{q['ar']}|{q['variant']}|{q['prediction_mse']:.6f}|"+'|'.join(f"{q['source_mse'][k]:.6f}" for k in ('T','M','J'))+f"|{q['delta_rmse']:.6f}|\n"
 return text
text='''# R6 创新残差似然与约束历史公式：实际训练报告
72新配置，种子1429/1543/1663，四段各1200。所有保留段结果现为开发信息；没有保留选模。旧R5不重训。

结构由当前时间基T、协变量一次/平方M、乘积J联合估计；f0=b0+bt*t/3600+bl*y_prev在base段冻结。生成层c+gT+gM+gJ，转换残差结构bias=c(1-bl)-b0、T=gT_cur-bl*gT_prev-bt*t/3600，M/J同过滤；filtered直接拟合该过滤残差。

新增协方差在结构段初始OLS残差上用精确Gaussian创新NLL，rho∈[.001,.975]、v/g0,R/g0∈[1e-6,4]；单起点最多160次评估，35迭代；IID与固定BIC门槛，数值/边界/预算失败回退IID。它是残差ML，不是REML或联合ML。冻结后一次GLS。锚点等价候选共享clean协方差，避免数值优化误差破坏规范等价。

输入t、当前/过去x,z、已观测过去y与冻结参数，无真实rho/A/eps。e_prev=y_prev-g_hat_prev-mu，mu=mean_structure(y-g_hat)，a_prev为处理e_prev后的posterior，当前prior=rho*a_prev。独立H段比较first=h0+h1e、自由state=h0+h1e+h2a、fixed=rho*a-bl*e+h0，仅拟合h0。真实改变了估计与函数约束，非坐标改名。

新增H_state=(rho-bl)*a、H_remainder=-bl*(e-a)、H_calibration=H-H_state-H_remainder。fixed校准项为常数；自由模型校准项可时变。输出bias/T/M/J、structural_total、e_prev/a_prev、以上H分列、H、prediction_total，以及f0+y_hat和标签后unexplained。状态模型贡献不是自然AR/传感器误差识别。

令d=mu-c，准确恒等式r-S=e_t-bl*e_prev+(1-bl)d。fixed理想常数为(1-bl)d，不能混用GLS c与mu。metrics逐候选保存mu、c、差异、理想h0与实际h0；filtered不虚构生成截距。

## 实际预算与核验
'''
text+='实际主训练计数：'+str(line)+'。2592复合模型，各自保存加载；预检另有一个小样本优化调用，nfev='+str(json.loads((ROOT/'preflight_verification.json').read_text())['optimizer_preflight']['optimization']['nfev'])+'。相同训练样本，额外计算不等于相同计算预算。\n'
for name in ('verification','independent_verification','likelihood_independent_verification'):
 q=json.loads((OUT/(name+'.json')).read_text());text+=name+': passed='+str(q['passed'])+'。\n'
text+='''白化与密集Gaussian NLL、posterior索引、显式原始权重、H恒等式、单点批量、未来标签/协变量隔离、冻结映射、原始/中心化闭合、相同四点过去固定、锚点规范、组内冗余、跨组零空间支持内和离支持均核验。独立审计禁止fit调用。

## 全开clean同数据对照（三种子均值）
'''
text+=table([q for q in agg if q['period']==q['interaction']==1 and '_clean_' in q['variant']])
text+='\n## 所有开关clean整体对照\n|设计|AR|估计器|state MSE|fixed MSE|fixed胜state/12|\n|---|---:|---|---:|---:|---:|\n'
for d in ('regular','correlated','overlap'):
 for ar in (0,1):
  ids=sorted({q['id'] for q in rows if q['design']==d and q['ar']==ar})
  for est in ('generation','filtered','gls','likelihood'):
   a=[lookup[i,est+'_clean_state'] for i in ids];b=[lookup[i,est+'_clean_fixed'] for i in ids]
   text+=f'|{d}|{ar}|{est}|{mean(a,lambda q:q["prediction_mse"]):.6f}|{mean(b,lambda q:q["prediction_mse"]):.6f}|'+str(sum(y['prediction_mse']<x['prediction_mse']-1e-10 for x,y in zip(a,b)))+'/12|\n'
text+='\n## 协方差选择与失败（每规范36配置）\n|规范|AR|方法|接受AR|回退原因|实际nfev|\n|---|---:|---|---:|---|---:|\n'
for spec in ('clean','underfit','anchor'):
 for ar in (0,1):
  for est in ('gls','likelihood'):
   qq=[q for q in rows if q['variant']==f'{est}_{spec}_state' and q['ar']==ar];c=collections.Counter(q['covariance']['fallback'] for q in qq);text+=f'|{spec}|{ar}|{est}|{c[None]}|'+str({k:v for k,v in c.items() if k is not None})+'|'+str(sum(q['covariance'].get('optimization',{}).get('nfev',0) for q in qq) if est=='likelihood' and spec!='anchor' else 0)+'|\n'
text+='''
无真实AR的漏M平方残差可以有真实序列相关，不一概称纯噪声误报。Gaussian AR协变量遗漏平方项协方差正比phi^(2h)。创新似然选AR不表明来源正确。

## 强相关无周期有乘积有AR：必须同时看源失败
'''
text+=table([q for q in agg if q['design']=='correlated' and (q['period'],q['interaction'],q['ar'])==(0,1,1) and q['variant'].endswith('_state')])
text+='\n## 原始与训练中心化生成源：likelihood clean state全开\n|设计|AR|T raw/centered|M raw/centered|J raw/centered|\n|---|---:|---|---|---|\n'
for d in ('regular','correlated','overlap'):
 for ar in (0,1):
  qq=[q for q in rows if q['design']==d and q['ar']==ar and q['period']==q['interaction']==1 and q['variant']=='likelihood_clean_state'];text+=f'|{d}|{ar}|'+ '|'.join('/'.join(f'{mean(qq,lambda q:q["generation_sources"][kind][k]["mse"]):.6f}' for kind in ('raw','centered')) for k in ('T','M','J'))+'|\n'
text+='\n## 状态与代理诊断：likelihood clean state全开\n|设计|AR|代理污染MSE|state对真AR MSE|H对潜在oracle MSE|mu-c均值|\n|---|---:|---:|---:|---:|---:|\n'
for d in ('regular','correlated','overlap'):
 for ar in (0,1):
  qq=[q for q in rows if q['design']==d and q['ar']==ar and q['period']==q['interaction']==1 and q['variant']=='likelihood_clean_state'];text+=f'|{d}|{ar}|'+ '|'.join(f'{mean(qq,lambda q:q[k]["mse"]):.6f}' for k in ('proxy_pollution','state_AR_score','H_oracle'))+f'|{mean(qq,lambda q:q["mu_minus_c"]):.6f}|\n'
text+='''
代理真值为过去AR+noise减结构段真误差均值；state对未中心化真AR诊断含规范偏移；H潜在oracle使用不可观测真实过去AR/noise，不能当同观测信息集可达基准。metrics保存所有规范/开关/候选的源原始、训练中心化MSE/bias/RMS、生成层、代理、state/H、完整预测及固定历史四点。ALL_CONFIGURATIONS.md含全部864个三种子汇总；不能仅看主表。

## 训练稳定性与结论边界
train_stability.json为864结构的训练段设计诊断；conditional_train_source_variance使用已拟合协方差，不含其估计不确定性或错模型风险。秩亏协方差仅为所选伪逆规范，null敏感性更关键；不用于选惩罚或候选。近共线、漏基与组间零空间仍未修复。低预测MSE可与巨大源误差和响应失真同时存在；创新似然/H约束不自动解决内容归属。

三固定种子合成一步滚动，没有总体、显著性、因果、真实数据、物理测量误差或部署证明。Muse未联系；没有重发被拒payload。连续目标仍active。
'''
(ROOT/'REPORT.md').write_text(text);(ROOT/'ALL_CONFIGURATIONS.md').write_text('# R6 全部开关与候选（三固定种子均值）\n'+table(agg))
print('report and all configurations written')
