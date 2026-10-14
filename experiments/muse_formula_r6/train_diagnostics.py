"""Read-only conditional train design stability; no fit, no holdout selection."""
from run import *
def main():
 rows=json.loads((OUT/'metrics.json').read_text());output=[];covchecks=[]
 for r in rows:
  if r['history']!='first':continue
  m=StateHistory.load(OUT/r['id']/(r['variant']+'.npz'));s=m.structure
  t,y,x,src=dgp(r['seed'],r['period'],r['interaction'],r['design'],r['ar']);st=np.arange(1201,2401)
  bb=basis(t[st],x[st],m.anchor,m.underfit) if m.estimator!='filtered' else blocks(t,y,x,st,m.coef[2],anchor=m.anchor,underfit=m.underfit)
  z=[np.ones((len(st),1))]+[(bb[k]/s.states[k]['scale'])@s.states[k]['rotation'] for k in GROUPS];d=np.column_stack(z)
  # GLS covariance conditional on frozen fitted state/noise model. OLS sandwich uses same assumed Sigma.
  n=len(st);pp=m.params
  if m.estimator in ('gls','likelihood'):
   w=innovations(d,pp);pinv=np.linalg.pinv(w,rcond=1e-10);C=pinv@pinv.T
  else:
   ii=np.arange(n);cov=pp['state_variance']*pp['rho']**abs(ii[:,None]-ii[None,:])+pp['R']*np.eye(n)
   pinv=np.linalg.pinv(d,rcond=1e-10);C=pinv@cov@pinv.T
  _,sv,vh=np.linalg.svd(d,full_matrices=False);rank=int(sum(sv>1e-10*sv[0]));null=vh[rank:].T
  pos=1;diagnostics={}
  for k,zk in zip(GROUPS,z[1:]):
   end=pos+zk.shape[1];ambiguity=float(np.linalg.norm(zk@null[pos:end])/np.sqrt(n)) if null.size else 0.
   diagnostics[k]=dict(conditional_train_source_variance=float(np.sum((zk@C[pos:end,pos:end])*zk)/n),train_null_sensitivity=ambiguity,unique_on_training_support=ambiguity<1e-8)
   pos=end
  output.append(dict(id=r['id'],variant=r['variant'],rank=rank,columns=d.shape[1],condition=r['certificate']['condition'],groups=diagnostics,interpretation='conditional assumed covariance; fitted covariance uncertainty and misspecification omitted; rank deficient covariance is chosen pseudoinverse gauge'))
  if m.estimator=='likelihood' and r['spec']!='anchor':
   op=pp['optimization'];assert op['nfev']<=160
   if op['nfev']:
    g=StateHistory.load(OUT/r['id']/('generation_'+r['spec']+'_first.npz'));res=y[st]-g.signal(t[st],x[st])-g.structure.weight[0];res-=res.mean();g0=np.mean(res**2);theta=op['theta'];rho,v,R=theta[0],theta[1]*g0,theta[2]*g0
    params=dict(rho=rho,state_variance=v,R=R,Q=v*(1-rho*rho));we=innovations(res,params);P=v;ld=0.
    for _ in res:
     F=P+R;ld+=np.log(F);P=rho*rho*(1-P/F)*P+params['Q']
    nn=.5*(len(res)*np.log(2*np.pi)+ld+we@we);error=abs(nn-op['ar_nll']);assert error<1e-7
    if pp['fallback'] is None:
     assert op['success'] and not op['boundary'] and 2*(op['iid_nll']-op['ar_nll'])>2*np.log(len(st))
    else:assert pp['rho']==pp['state_variance']==pp['Q']==0
    covchecks.append(dict(id=r['id'],spec=r['spec'],nll_error=error,nfev=op['nfev'],fallback=pp['fallback']))
 dump(OUT/'train_stability.json',output);dump(OUT/'likelihood_independent_verification.json',dict(passed=True,no_fit=True,fits=len(covchecks),checks=covchecks))
 print('train-only stability and likelihood audits',len(output),len(covchecks))
if __name__=='__main__':main()
