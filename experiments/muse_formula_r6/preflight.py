from model import *
from pathlib import Path
params=dict(rho=.7,state_variance=.4,Q=.4*(1-.7**2),R=.08)
n=20;ii=np.arange(n);cov=.4*.7**abs(ii[:,None]-ii[None,:])+.08*np.eye(n);W=innovations(np.eye(n),params)
err=float(np.max(abs(W@cov@W.T-np.eye(n))));assert err<1e-10
for e,reason in [(np.zeros(20),'zero_variance'),(np.tile([1.,0.,-1.,0.],20),'weak_lag1')]:
 assert estimate_covariance(e)['fallback']==reason
# Independent conditional mean of A_last from finite noisy observations.
e=np.linspace(-.7,.8,n);a=0.;P=.4
for value in e:
 K=P/(P+.08);a=a+K*(value-a);last=a;a=.7*a;P=.7**2*(1-K)*P+params['Q']
expect=(.4*.7**(n-1-ii))@np.linalg.solve(cov,e);assert abs(last-expect)<1e-10
rng=np.random.default_rng(91);a=rng.normal(size=100);z=np.zeros(100)
s=JointStructural().fit(dict(T=np.column_stack((a,a,z)),M=a[:,None],J=z[:,None]),a,z)
assert s.certificate['group_ranks']==dict(T=1,M=1,J=0) and not s.certificate['identifiable']
Path(__file__).with_name('preflight_verification.json').write_text(json.dumps(dict(passed=True,whitening_identity_max=err,posterior_dense_check=abs(last-expect),fallback_cases=True,group_redundancy=s.certificate),indent=2)+'\n')
print('whitening, state recursion, fallback, redundant-group algebra passed')

# Innovation likelihood matches independent dense Gaussian NLL.
wh=innovations(e,params);P=params['state_variance'];logdet=0.
for i in range(n):
 F=P+params['R'];logdet+=np.log(F);P=params['rho']**2*(1-P/F)*P+params['Q']
nll=.5*(n*np.log(2*np.pi)+logdet+wh@wh)
dense=.5*(n*np.log(2*np.pi)+np.linalg.slogdet(cov)[1]+e@np.linalg.solve(cov,e))
assert abs(nll-dense)<1e-10
lp=likelihood_covariance(np.zeros(20),estimate_covariance(np.zeros(20)))
assert lp['optimization']['nfev']==0 and lp['fallback']=='zero_variance'
# One independent synthetic optimizer call, distinct from primary training data.
rng=np.random.default_rng(919);a=0.;es=[]
for i in range(400):
 a=.7*a+rng.normal(0,.4);es.append(a+rng.normal(0,.25))
lp=likelihood_covariance(es,estimate_covariance(es));assert lp['optimization']['nfev']<=160
p=Path(__file__).with_name('preflight_verification.json');q=json.loads(p.read_text());q.update(innovation_nll_dense_error=abs(nll-dense),optimizer_preflight_calls=1,optimizer_preflight=lp);p.write_text(json.dumps(q,indent=2)+'\n')
print('dense innovation NLL, optimizer budget and zero residual fallback passed')
