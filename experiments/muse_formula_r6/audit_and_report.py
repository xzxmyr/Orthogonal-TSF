"""Read-only independent expansion and audits, no regressions."""
from run import *
def main():
 rows=json.loads((OUT/'metrics.json').read_text());checks=[];witness=[];params_rows=[]
 # Any accidental fit in this audit is an error.
 def forbidden(*a,**k):raise AssertionError('audit attempted training')
 JointStructural.fit=forbidden;StateHistory.fit_history=forbidden
 idx=np.arange(1,4801);st=np.arange(1201,2401);te=np.arange(3601,4801);train=st-1;test=te-1
 for ident in sorted({r['id'] for r in rows}):
  rr=[r for r in rows if r['id']==ident];q=rr[0];t,y,x,src=dgp(q['seed'],q['period'],q['interaction'],q['design'],q['ar'])
  with np.load(OUT/ident/'points.npz',allow_pickle=False) as saved:
   assert np.array_equal(saved['y'],y[idx])
   for pref in ('truth_','centered_truth_'):assert np.max(abs(saved['r']-sum(saved[pref+k] for k in ('bias','T','M','J','AR','noise'))))<1e-8
   for row in rr:
    name=row['variant'];m=StateHistory.load(OUT/ident/(name+'.npz'));stamp=fingerprint(m);s=m.structure;out=m.predict(t,y,x,idx)
    err=max(float(np.max(abs(out[k]-saved[name+'_'+k]))) for k in out);assert err<1e-8
    # Independent raw-weight expansion, including original-scale matrices.
    b=basis(t,x,m.anchor,m.underfit);start=1;signals={}
    for k in GROUPS:
     state=s.states[k];size=state['rotation'].shape[1];raw_w=(state['rotation']@s.weight[start:start+size])/state['scale'];start+=size;signals[k]=b[k]@raw_w
    signal=sum(signals.values());ep=y[idx-1]-signal[idx-1]-m.mu
    assert np.max(abs(ep-out['e_prev']))<1e-8
    c=s.weight[0]*(1-m.coef[2])-m.coef[0] if m.estimator!='filtered' else s.weight[0]
    assert np.max(abs(c-out['bias']))<1e-8
    for k in GROUPS:
     expected=signals[k][idx]-m.coef[2]*signals[k][idx-1]
     if k=='T':expected-=m.coef[1]*t[idx]/3600
     assert np.max(abs(expected-out[k]))<1e-8
    # Independent scalar state update after e_prev, never after e_current.
    ap=np.zeros(len(t));a=0.;P=m.params['state_variance'];rho=m.params['rho'];e=y-signal-m.mu
    for pos in range(m.start,4800):
     K=P/max(P+m.params['R'],1e-12);posterior=(1-K)*a+K*e[pos];ap[pos]=posterior;a=rho*posterior;P=rho*rho*(1-K)*P+m.params['Q']
    assert np.max(abs(ap[idx-1]-out['a_prev']))<1e-8
    if m.history=='fixed':expected_h=m.h_weight[0]+rho*ap[idx-1]-m.coef[2]*ep
    else:
     expected_h=m.h_weight[0]+m.h_weight[1]*ep
     if m.history in ('state','fixed'):expected_h+=m.h_weight[2]*ap[idx-1]
    assert np.max(abs(out['H_state']-(rho-m.coef[2])*ap[idx-1]))<1e-8
    assert np.max(abs(out['H_remainder']+m.coef[2]*(ep-ap[idx-1])))<1e-8
    assert np.max(abs(out['H']-out['H_state']-out['H_remainder']-out['H_calibration']))<1e-8
    assert np.max(abs(expected_h-out['H']))<1e-8
    assert np.max(abs(saved[name+'_y_hat']-saved['f0']-out['prediction_total']))<1e-8
    assert np.max(abs(saved['y']-saved[name+'_y_hat']-saved[name+'_unexplained']))<1e-8
    assert np.max(abs(sum(saved[name+'_centered_'+k] for k in ('bias',)+GROUPS)-out['structural_total']))<1e-8
    for pos in (0,37,1199):
     oneidx=te[pos:pos+1];one=m.predict(t,y,x,oneidx);assert max(abs(one[k][0]-out[k][test[pos]]) for k in one)<1e-8
     yf=y.copy();xf=x.copy();yf[oneidx[0]:]+=17000;xf[oneidx[0]+1:]+=7000
     future=m.predict(t,yf,xf,oneidx);assert max(abs(one[k][0]-future[k][0]) for k in one)<1e-8
    # GLS map was trained without holdout: frozen model bytes/parameters unchanged.
    assert fingerprint(m)==stamp
    if m.estimator!='filtered' and m.history=='first':
     gt=dict(bias=np.full(4800,12-.6*q['interaction']*m.anchor[0]*m.anchor[1]),T=src['T'][idx],M=src['M'][idx]+.6*q['interaction']*(m.anchor[1]*x[idx,0]+m.anchor[0]*x[idx,1]),J=.6*q['interaction']*(x[idx,0]-m.anchor[0])*(x[idx,1]-m.anchor[1]))
     go=dict(bias=np.full(4800,s.weight[0]),**{k:signals[k][idx] for k in GROUPS});gc=centered(go,train);gct=centered(gt,train)
     assert np.max(abs(y[idx]-sum(gct.values())-src['AR'][idx]-src['noise'][idx]))<1e-8
     for kind,estimate,truth in [('raw',go,gt),('centered',gc,gct)]:
      for k in ('bias',)+GROUPS:
       v=score(estimate[k][test],truth[k][test]);assert abs(v['mse']-row['generation_sources'][kind][k]['mse'])<1e-8
    if m.history in ('state','fixed'):params_rows.append(dict(id=ident,variant=name,params=m.params,history_feature_rank=int(np.linalg.matrix_rank(np.column_stack((np.ones(1200),out['e_prev'][ht_indices:=np.arange(2400,3600)],out['a_prev'][ht_indices])))),true_rho_diagnostic=.8 if q['ar'] else 0.))
    if q['design']=='overlap' and q['period']==q['interaction']==1 and row['spec']=='clean' and m.history in ('state','fixed'):
     bb=basis(t[st],x[st]) if m.estimator!='filtered' else blocks(t,y,x,st,m.coef[2]);dd=np.column_stack([np.ones(len(st))]+[(bb[k]/s.states[k]['scale'])@s.states[k]['rotation'] for k in GROUPS]);_,_,v=np.linalg.svd(dd,full_matrices=False)
     alt=StateHistory.load(OUT/ident/(name+'.npz'));alt.structure.weight+=10*v[-1];alt.mu=float(np.mean(y[st]-alt.signal(t[st],x[st])));oo=alt.predict(t,y,x,te);base={k:out[k][test] for k in out}
     on=max(float(np.max(abs(oo[k]-base[k]))) for k in ('structural_total','prediction_total','H','a_prev','e_prev'));assert on<1e-7
     diffs=[]
     for dx,dz in ((0,0),(.5,0),(0,.4),(.5,.4)):
      aout=alt.predict(t,y,x,te,x[te]+[dx,dz]);bout=m.predict(t,y,x,te,x[te]+[dx,dz]);diffs.append(aout['prediction_total']-bout['prediction_total'])
     witness.append(dict(id=ident,variant=name,rank=s.certificate['rank'],columns=s.certificate['columns'],on_support_max=on,off_support_x_shift_rms=float(np.sqrt(np.mean(diffs[1]**2))),mixed_difference_max=float(np.max(abs(diffs[3]-diffs[1]-diffs[2]+diffs[0]))),component_rms={k:score(oo[k],base[k])['rmse'] for k in ('bias',)+GROUPS},null_vector=(10*v[-1]).tolist()))
    checks.append(dict(id=ident,variant=name,reload_max=err,expanded_sources=True,expanded_state=True,three_singletons=True,future_y_and_x=True,raw_and_centered_closure=True,frozen=True))
  print('audit',ident,flush=True)
 dump(OUT/'independent_verification.json',dict(passed=True,no_fit_calls=True,models=len(checks),checks=checks));dump(OUT/'off_support_null_witness.json',witness);dump(OUT/'covariance_diagnostics.json',params_rows)
 agg=[]
 for design,ar,p,j,name in itertools.product(['regular','correlated','overlap'],[0,1],[0,1],[0,1],sorted({r['variant'] for r in rows})):
  rr=[r for r in rows if (r['design'],r['ar'],r['period'],r['interaction'],r['variant'])==(design,ar,p,j,name)];avg=lambda f:sum(f(r) for r in rr)/len(rr)
  agg.append(dict(design=design,ar=ar,period=p,interaction=j,variant=name,structural_mse=avg(lambda r:r['structural_mse']),prediction_mse=avg(lambda r:r['prediction_mse']),gain=avg(lambda r:r['prediction_gain']),proxy_mse=avg(lambda r:r['proxy_pollution']['mse']),state_mse=avg(lambda r:r['state_AR_score']['mse']),H_oracle_mse=avg(lambda r:r['H_oracle']['mse']),source_mse={k:avg(lambda r:r['sources'][k]['mse']) for k in GROUPS},delta_rmse=avg(lambda r:r['mixed_response']['J']['rmse'])))
 dump(OUT/'aggregate_metrics.json',agg);print('2592 independent model audits passed; null witnesses',len(witness))
if __name__=='__main__':main()
