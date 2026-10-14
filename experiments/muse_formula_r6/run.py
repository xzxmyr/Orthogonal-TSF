from pathlib import Path
import json,hashlib,itertools
import numpy as np
from model import *
ROOT=Path(__file__).resolve().parent;OUT=ROOT/"artifacts"
def dump(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def score(a,b):
 a=np.asarray(a);b=np.asarray(b);err=a-b
 return dict(mse=float(np.mean(err*err)),rmse=float(np.sqrt(np.mean(err*err))),bias=float(np.mean(err)),estimate_rms=float(np.sqrt(np.mean(a*a))),truth_rms=float(np.sqrt(np.mean(b*b))),corr=None if min(np.std(a),np.std(b))<1e-12 else float(np.corrcoef(a,b)[0,1]))
def dgp(seed,p,j,design,ar):
 rng=np.random.default_rng(seed);t=np.arange(4801);uv=rng.normal(0,.7,(len(t),2));u=np.zeros((len(t),2))
 for i in range(1,len(t)):u[i]=.55*u[i-1]+uv[i]
 si=np.sin(2*np.pi*t/60);co=np.cos(2*np.pi*t/144)
 x=2+.8*p*si+u[:,0];z=-1+.5*p*co+u[:,1]
 if design=='correlated':z=-1+.5*p*co+.95*u[:,0]+.05*u[:,1]
 if design=='overlap':x=2+si
 x=np.column_stack((x,z));eta=rng.normal(0,.4,len(t));A=np.zeros(len(t))
 for i in range(1,len(t)):A[i]=.8*A[i-1]+eta[i]
 src=dict(T=1.2*t/3600+p*(2*si+.8*co),M=.7*x[:,0]+.35*x[:,0]**2-.5*x[:,1]+.2*x[:,1]**2,J=.6*j*x[:,0]*x[:,1],AR=ar*A,noise=rng.normal(0,.25,len(t)))
 return t,12+sum(src.values()),x,src

def ledger(src,coef,idx,t,x,j,anchor=(0.,0.)):
 s={k:v.copy() for k,v in src.items()};a,b=anchor;c=12-.6*j*a*b
 s['J']=.6*j*(x[:,0]-a)*(x[:,1]-b);s['M']+=.6*j*(b*x[:,0]+a*x[:,1])
 o={k:v[idx]-coef[2]*v[idx-1] for k,v in s.items()};o['T']-=coef[1]*t[idx]/3600
 o['bias']=np.full(len(idx),c*(1-coef[2])-coef[0]);return o

def centered(d,train):
 o={k:v-v[train].mean() for k,v in d.items() if k!='bias'};o['bias']=d['bias']+sum(v[train].mean() for k,v in d.items() if k!='bias');return o

SPECS={'clean':{},'underfit':{'underfit':True},'anchor':{'anchor':(2.,-1.)}}
def fingerprint(m):
    return hashlib.sha256(b''.join(a.tobytes() for a in [m.structure.weight,m.coef,m.h_weight,np.array(m.mu)]+[v for st in m.structure.states.values() for v in st.values()])+json.dumps(m.params,sort_keys=True).encode()).hexdigest()
def main():
 if OUT.exists():raise RuntimeError('R6 is single execution; do not overwrite any artifacts')
 OUT.mkdir();cfg=dict(seeds=[1429,1543,1663],designs=['regular','correlated','overlap'],period=[0,1],interaction=[0,1],ar=[0,1],specs=SPECS,estimators=['generation','filtered','gls','likelihood'],history=['first','state','fixed'],split={'base':[1,1200],'structure':[1201,2400],'history':[2401,3600],'holdout':[3601,4800]},sha={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'model.py',ROOT/'run.py',ROOT/'PROTOCOL_FROZEN.md',ROOT/'FORMULA_REVIEW.md']})
 cfg['likelihood']={'method':'innovation residual ML (not REML)','bounds':[[.001,.975],[1e-6,4.],[1e-6,4.]],'maxiter':35,'maxfun':160,'maxls':10,'IID_selection':'AR requires 2*(IID_NLL-AR_NLL)>2*log(n)','anchor_reuses_clean_covariance':True};dump(OUT/'config.json',cfg);rows=[];checks=[];lineage=dict(base=0,generation_ols=0,filtered_ols=0,gls=0,likelihood_gls=0,likelihood_fit=0,likelihood_nfev=0,H=0)
 idx=np.arange(1,4801);st=np.arange(1201,2401);ht=np.arange(2401,3601);te=np.arange(3601,4801);train=st-1;test=te-1
 for seed,design,p,j,ar in itertools.product(cfg['seeds'],cfg['designs'],cfg['period'],cfg['interaction'],cfg['ar']):
  ident=f's{seed}_{design}_p{p}_j{j}_ar{ar}';folder=OUT/ident;folder.mkdir();t,y,x,src=dgp(seed,p,j,design,ar)
  d=np.column_stack((np.ones(4800),t[idx]/3600,y[idx-1]));coef=np.linalg.lstsq(d[:1200],y[1:1201],rcond=1e-12)[0];lineage['base']+=1
  np.savez_compressed(folder/'frozen_base.npz',coef=coef);f0=d@coef;r=y[idx]-f0;truth=ledger(src,coef,idx,t,x,j);ct=centered(truth,train)
  assert np.max(np.abs(r-sum(truth.values())))<1e-8
  arr=dict(t=t[idx],x=x[idx],y=y[idx],f0=f0,r=r,trend=-coef[1]*t[idx]/3600,filtered_T=truth['T']+coef[1]*t[idx]/3600)
  arr.update({'truth_'+k:v for k,v in truth.items()});arr.update({'centered_truth_'+k:v for k,v in ct.items()});outs={}
  likelihood_clean=None
  for spec,opts in SPECS.items():
   anchor=opts.get('anchor',(0.,0.));under=opts.get('underfit',False);bb=basis(t[st],x[st],anchor,under)
   gen=JointStructural().fit(bb,y[st],np.zeros(len(st)));lineage['generation_ols']+=1
   gres=y[st]-gen.predict(bb,np.zeros(len(st)))['total'];gp=estimate_covariance(gres)
   gls=feasible_gls(gen,bb,y[st],gp);lineage['gls']+=1
   if spec=='anchor':lp=likelihood_clean
   else:
    lp=likelihood_covariance(gres,gp);lineage['likelihood_fit']+=1;lineage['likelihood_nfev']+=lp['optimization']['nfev']
    if spec=='clean':likelihood_clean=lp
   lik=feasible_gls(gen,bb,y[st],lp);lineage['likelihood_gls']+=1
   fil=JointStructural().fit(blocks(t,y,x,st,coef[2],anchor=anchor,underfit=under),r[train],coef[1]*t[st]/3600);lineage['filtered_ols']+=1
   fg=fil.predict(bb,np.zeros(len(st)));fp=estimate_covariance(y[st]-sum(fg[k] for k in GROUPS))
   for estimator,s,params in [('generation',gen,gp),('filtered',fil,fp),('gls',gls,gp),('likelihood',lik,lp)]:
    frozen=s.weight.copy()
    for history in ('first','state','fixed'):
     name=f'{estimator}_{spec}_{history}';m=StateHistory(s,coef,params,estimator,history,**opts).fit_history(t,y,x,st,ht);lineage['H']+=1
     assert np.array_equal(s.weight,frozen);m.save(folder/(name+'.npz'));o=m.predict(t,y,x,idx);outs[name]=o
     loaded=StateHistory.load(folder/(name+'.npz'));stamp=fingerprint(loaded);oo=loaded.predict(t,y,x,idx);err=max(float(np.max(abs(o[k]-oo[k]))) for k in o);assert err<1e-8
     one=loaded.predict(t,y,x,te[37:38]);assert max(abs(one[k][0]-o[k][test[37]]) for k in one)<1e-8
     yf=y.copy();yf[te[37]:]+=10000;future=loaded.predict(t,yf,x,te[37:38]);assert max(abs(one[k][0]-future[k][0]) for k in one)<1e-8
     chosen=ledger(src,coef,idx,t,x,j,anchor);cchosen=centered(chosen,train);cout=centered({k:o[k] for k in ('bias',)+GROUPS},train)
     proxy=src['AR'][idx-1]+src['noise'][idx-1]-np.mean(src['AR'][st]+src['noise'][st]);oracle=(.8-coef[2])*src['AR'][idx-1]-coef[2]*src['noise'][idx-1]
     row=dict(id=ident,seed=seed,design=design,period=p,interaction=j,ar=ar,variant=name,estimator=estimator,spec=spec,history=history,base_mse=score(r[test],0)['mse'],structural_mse=score(o['structural_total'][test],r[test])['mse'],prediction_mse=score(o['prediction_total'][test],r[test])['mse'],sources={k:score(o[k][test],chosen[k][test]) for k in ('bias',)+GROUPS},centered_sources={k:score(cout[k][test],cchosen[k][test]) for k in ('bias',)+GROUPS},proxy_pollution=score(o['e_prev'][test],proxy[test]),state_AR_score=score(o['a_prev'][test],src['AR'][te-1]),H_oracle=score(o['H'][test],oracle[test]),certificate=s.certificate,covariance=params,h_weight=m.h_weight.tolist(),mu=m.mu,mu_minus_c=float(m.mu-s.weight[0]) if estimator!='filtered' else None,ideal_fixed_intercept=float((1-coef[2])*(m.mu-s.weight[0])) if estimator!='filtered' else None,H_parts={k:score(o[k][test],0) for k in ('H_state','H_remainder','H_calibration')},generation_constant=float(s.weight[0]) if estimator!='filtered' else None)
     if estimator!='filtered':
      go=s.predict(basis(t[idx],x[idx],anchor,under),np.zeros(len(idx)));a,b=anchor
      gt=dict(bias=np.full(len(idx),12-.6*j*a*b),T=src['T'][idx],M=src['M'][idx]+.6*j*(b*x[idx,0]+a*x[idx,1]),J=.6*j*(x[idx,0]-a)*(x[idx,1]-b));gc=centered({k:go[k] for k in ('bias',)+GROUPS},train);gct=centered(gt,train)
      row['generation_sources']={'raw':{k:score(go[k][test],gt[k][test]) for k in ('bias',)+GROUPS},'centered':{k:score(gc[k][test],gct[k][test]) for k in ('bias',)+GROUPS}}
      assert np.max(abs(y[idx]-sum(gt.values())-src['AR'][idx]-src['noise'][idx]))<1e-8
     ivs=[]
     for dx,dz in ((0,0),(.5,0),(0,.4),(.5,.4)):
      current=x[te]+[dx,dz];oi=loaded.predict(t,y,x,te,current);oi1=loaded.predict(t,y,x,te[37:38],current[37:38]);assert max(abs(oi1[k][0]-oi[k][37]) for k in oi)<1e-8
      for k in ('H','a_prev','e_prev'):assert np.max(abs(oi[k]-o[k][test]))<1e-8
      fi=coef[0]+coef[1]*t[te]/3600+coef[2]*y[te-1];yi=12+src['T'][te]+src['AR'][te]+src['noise'][te]+.7*current[:,0]+.35*current[:,0]**2-.5*current[:,1]+.2*current[:,1]**2+.6*j*current[:,0]*current[:,1];ivs.append((yi-fi,oi))
     delta=lambda a:a[3]-a[1]-a[2]+a[0];true=delta([a[0] for a in ivs]);assert np.max(abs(true-.12*j))<1e-8
     row['mixed_response']={}
     for k in ('structural_total','prediction_total','T','M','J','H'):
      est=delta([a[1][k] for a in ivs]);v=score(est,true if k in ('structural_total','prediction_total','J') else np.zeros(len(te)));v['corr']=None;row['mixed_response'][k]=v
     row['prediction_gain']=row['structural_mse']-row['prediction_mse']
     arr.update({name+'_'+k:v for k,v in o.items()});arr[name+'_y_hat']=f0+o['prediction_total'];arr[name+'_unexplained']=r-o['prediction_total'];arr.update({name+'_centered_'+k:v for k,v in cout.items()})
     assert np.max(abs(o['structural_total']-sum(o[k] for k in ('bias',)+GROUPS)))<1e-8;assert fingerprint(loaded)==stamp
     if design=='overlap':assert not s.certificate['identifiable']
     if design=='regular':assert s.certificate['identifiable']
     rows.append(row);checks.append(dict(id=ident,variant=name,reload_max=err,singleton=True,future_labels=True,structure_frozen=True,parameters_frozen=True,closure=True,history_fixed_intervention=True))
  for estimator,history in itertools.product(('generation','filtered','gls','likelihood'),('first','state','fixed')):
   clean=outs[f'{estimator}_clean_{history}'];anchor_o=outs[f'{estimator}_anchor_{history}'];ae=max(float(np.max(abs(clean[k]-anchor_o[k]))) for k in ('structural_total','prediction_total','e_prev','a_prev','H'));assert ae<1e-7
   checks.append(dict(id=ident,estimator=estimator,history=history,anchor_equivalence=ae))
  np.savez_compressed(folder/'points.npz',**arr);dump(OUT/'execution_lineage.json',lineage);print(ident,flush=True)
 dump(OUT/'metrics.json',rows);dump(OUT/'verification.json',dict(passed=True,configs=72,structures=864,history_models=len(rows),checks=checks));print('R6 completed',lineage,flush=True)
if __name__=='__main__':main()
