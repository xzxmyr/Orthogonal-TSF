"""R6 bounded innovation likelihood and constrained history; frozen SVD maps."""
import json
import numpy as np

GROUPS = ('T', 'M', 'J')

def basis(t, x, anchor=(0., 0.), underfit=False):
    T = np.column_stack((t/3600, np.sin(2*np.pi*t/60), np.cos(2*np.pi*t/60),
                         np.sin(2*np.pi*t/144), np.cos(2*np.pi*t/144)))
    M = x if underfit else np.column_stack((x, x*x))
    J = ((x[:, 0]-anchor[0])*(x[:, 1]-anchor[1]))[:, None]
    return dict(T=T, M=M, J=J)

def blocks(t, y, x, idx, bl, mode='filtered', anchor=(0.,0.), underfit=False, history=False, current_x=None):
    cur = basis(t[idx], x[idx] if current_x is None else current_x, anchor, underfit)
    prev = basis(t[idx-1], x[idx-1], anchor, underfit)
    if mode == 'ordered':
        out = {k:np.column_stack((cur[k],prev[k])) for k in GROUPS}
    else:
        out = {k:cur[k]-bl*prev[k] if mode=='filtered' else cur[k] for k in GROUPS}
    if history or mode=='ordered':
        out['T'] = np.column_stack((out['T'], y[idx-1]))
    return out

class JointStructural:
    def fit(self, blocks, target, trend):
        self.states = {}
        design = [np.ones(len(target))]
        for k in GROUPS:
            x = np.asarray(blocks[k], float)
            scale = np.sqrt(np.mean(x*x, axis=0)); scale[scale<1e-12]=1
            _, s, v = np.linalg.svd(x/scale, full_matrices=False)
            keep=s>1e-10*np.sqrt(len(x))
            rotation=v[keep].T
            self.states[k] = dict(scale=scale, rotation=rotation)
            design.append((x/scale)@rotation)
        d=np.column_stack(design)
        self.weight=np.linalg.lstsq(d, np.asarray(target)+trend, rcond=1e-10)[0]
        s=np.linalg.svd(d, compute_uv=False)
        rank=int(np.sum(s>1e-10*s[0]))
        self.certificate=dict(columns=d.shape[1], rank=rank, identifiable=rank==d.shape[1],
            condition=float(s[0]/s[-1]) if rank==d.shape[1] else None,
            nonzero_condition=float(s[0]/s[rank-1]), singular_values=s.tolist(),
            group_ranks={k:self.states[k]['rotation'].shape[1] for k in GROUPS})
        return self

    def predict(self, blocks, trend):
        out={'bias':np.full(len(trend),self.weight[0])};start=1
        for k in GROUPS:
            st=self.states[k];z=(blocks[k]/st['scale'])@st['rotation']
            out[k]=z@self.weight[start:start+z.shape[1]];start+=z.shape[1]
        out['T']=out['T']-trend
        out['total']=sum(out[k] for k in ('bias',)+GROUPS)
        return out

    def save(self,path):
        arrays={'weight':self.weight,'certificate':np.array(json.dumps(self.certificate))}
        arrays.update({k+'_'+n:v for k,st in self.states.items() for n,v in st.items()})
        np.savez_compressed(path,**arrays)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            obj=cls();obj.weight=d['weight'].copy();obj.certificate=json.loads(str(d['certificate']))
            obj.states={k:{n:d[k+'_'+n].copy() for n in ('scale','rotation')} for k in GROUPS}
        return obj

def estimate_covariance(residual):
    e=np.asarray(residual)-np.mean(residual)
    gamma=[float(np.mean(e[k:]*e[:len(e)-k])) for k in range(3)]
    g0,g1,g2=gamma; reason=None;rho=0.;v=0.;R=max(g0,1e-12)
    if g0<=1e-12:reason='zero_variance'
    elif abs(g1)<=.05*g0:reason='weak_lag1'
    else:
        rr=g2/g1; vv=g1/rr if abs(rr)>1e-12 else -1.;r=g0-vv
        if not (0<rr<.98):reason='unstable_or_nonpositive_rho'
        elif vv<=0 or r<0:reason='negative_variance'
        else:rho=rr;v=vv;R=r
    return dict(rho=rho,state_variance=v,Q=v*(1-rho*rho),R=R,gamma=gamma,fallback=reason)

def likelihood_covariance(residual, moment):
    """Conditional Gaussian innovation ML on train-only OLS residuals (not REML).
    Covariance is frozen before one GLS update; not joint/profile structural ML.
    """
    from scipy.optimize import minimize
    e=np.asarray(residual,float);e=e-e.mean();g0=float(np.mean(e*e))
    if g0<=1e-12:
        return dict(moment,method='innovation_residual_ml',optimization={'reason':'zero_variance','nfev':0})
    def unpack(z):
        rho=float(z[0]);v=float(z[1]*g0);R=float(z[2]*g0)
        return dict(rho=rho,state_variance=v,Q=v*(1-rho*rho),R=R)
    calls=0;last_z=None;last_value=None
    class BudgetLimit(Exception):pass
    def objective(z):
        nonlocal calls,last_z,last_value
        if calls>=160:raise BudgetLimit()
        calls+=1
        p=unpack(z);a=0.;P=p['state_variance'];value=0.
        for y in e:
            F=P+p['R'];v=y-a;value+=np.log(F)+v*v/F
            K=P/F;a=p['rho']*(a+K*v);P=p['rho']**2*(1-K)*P+p['Q']
        last_z=np.array(z);last_value=float(.5*(value+len(e)*np.log(2*np.pi)))
        return last_value
    iid=.5*len(e)*(np.log(2*np.pi*g0)+1)
    start=[moment['rho'],moment['state_variance']/g0,moment['R']/g0] if moment['fallback'] is None else [.5,.5,.5]
    bounds=[(.001,.975),(1e-6,4.),(1e-6,4.)]
    start=np.array([np.clip(v,lo+1e-5,hi-1e-5) for v,(lo,hi) in zip(start,bounds)])
    try:opt=minimize(objective,start,method='L-BFGS-B',bounds=bounds,options={'maxiter':35,'maxfun':160,'ftol':1e-9,'gtol':1e-5,'maxls':10})
    except (BudgetLimit,FloatingPointError,ValueError) as exc:
        from types import SimpleNamespace
        opt=SimpleNamespace(x=last_z if last_z is not None else start,fun=last_value if last_value is not None else iid,success=False,message=type(exc).__name__,nfev=calls,nit=0)
    boundary=any(abs(z-lo)<1e-5 or abs(z-hi)<1e-5 for z,(lo,hi) in zip(opt.x,bounds))
    reason=None
    if not opt.success or not np.isfinite(opt.fun):reason='numerical_or_budget_failure'
    elif boundary:reason='boundary_solution'
    elif 2*(iid-opt.fun)<=2*np.log(len(e)):reason='iid_bic'
    result=unpack(opt.x) if reason is None else dict(rho=0.,state_variance=0.,Q=0.,R=g0)
    result.update(gamma=moment['gamma'],fallback=reason,method='innovation_residual_ml',optimization=dict(success=bool(opt.success),message=str(opt.message),nfev=calls,nit=int(opt.nit),iid_nll=float(iid),ar_nll=float(opt.fun),bic_required_improvement=float(2*np.log(len(e))),boundary=boundary,theta=opt.x.tolist(),start=start.tolist()))
    return result

def innovations(values,params):
    """Linear causal whitening, stationary prior, applied equally to target/design."""
    values=np.asarray(values);out=np.empty_like(values,dtype=float)
    a=np.zeros(values.shape[1:]);P=params['state_variance'];rho=params['rho']
    for i,row in enumerate(values):
        F=max(P+params['R'],1e-12);v=row-a;out[i]=v/np.sqrt(F)
        K=P/F;a=rho*(a+K*v);P=rho*rho*(1-K)*P+params['Q']
    return out

def feasible_gls(initial,bb,target,params):
    obj=JointStructural();obj.states={k:{n:v.copy() for n,v in st.items()} for k,st in initial.states.items()}
    d=np.column_stack([np.ones(len(target))]+[(bb[k]/obj.states[k]['scale'])@obj.states[k]['rotation'] for k in GROUPS])
    wd=innovations(d,params);wy=innovations(target,params)
    obj.weight=np.linalg.lstsq(wd,wy,rcond=1e-10)[0]
    obj.certificate=dict(initial.certificate)
    ss=np.linalg.svd(wd,compute_uv=False);rank=int(np.sum(ss>1e-10*ss[0]))
    obj.certificate['whitened_rank']=rank
    obj.certificate['whitened_condition']=float(ss[0]/ss[-1]) if rank==wd.shape[1] else None
    return obj

class StateHistory:
    def __init__(self,structure,coef,params,estimator='generation',history='state',anchor=(0.,0.),underfit=False):
        self.structure=structure;self.coef=np.asarray(coef);self.params=params
        self.estimator=estimator;self.history=history;self.anchor=anchor;self.underfit=underfit
        self.start=1201
    def signal(self,t,x):
        o=self.structure.predict(basis(t,x,self.anchor,self.underfit),np.zeros(len(t)))
        return sum(o[k] for k in GROUPS)
    def structural(self,t,y,x,idx,current_x=None):
        o=self.structure.predict(blocks(t,y,x,idx,self.coef[2],anchor=self.anchor,underfit=self.underfit,current_x=current_x),self.coef[1]*t[idx]/3600)
        if self.estimator!='filtered':
            o['bias']=np.full(len(idx),self.structure.weight[0]*(1-self.coef[2])-self.coef[0])
            o['total']=sum(o[k] for k in ('bias',)+GROUPS)
        return o
    def history_features(self,t,y,x,idx):
        # Read labels strictly before max requested index, never current/future labels.
        last=int(max(idx));e=y[:last]-self.signal(t[:last],x[:last])-self.mu
        states=np.zeros(last);a=0.;P=self.params['state_variance'];rho=self.params['rho']
        for i in range(self.start,last):
            F=max(P+self.params['R'],1e-12);K=P/F
            a=a+K*(e[i]-a);states[i]=a
            a=rho*a;P=rho*rho*(1-K)*P+self.params['Q']
        ep=e[idx-1];ap=states[idx-1]
        features=np.column_stack((np.ones(len(idx)),ep)) if self.history=='first' else np.column_stack((np.ones(len(idx)),ep,ap))
        return features,ep,ap
    def fit_history(self,t,y,x,st,ht):
        self.mu=float(np.mean(y[st]-self.signal(t[st],x[st])))
        features,_,_=self.history_features(t,y,x,ht);o=self.structural(t,y,x,ht)
        f=self.coef[0]+self.coef[1]*t[ht]/3600+self.coef[2]*y[ht-1]
        target=y[ht]-f-o['total']
        if self.history=='fixed':
            core=self.params['rho']*features[:,2]-self.coef[2]*features[:,1]
            self.h_weight=np.array([np.mean(target-core)])
        else:self.h_weight=np.linalg.lstsq(features,target,rcond=1e-12)[0]
        return self
    def predict(self,t,y,x,idx,current_x=None):
        o=self.structural(t,y,x,idx,current_x);features,ep,ap=self.history_features(t,y,x,idx)
        o['structural_total']=o.pop('total');o['e_prev']=ep;o['a_prev']=ap
        o['H_state']=(self.params['rho']-self.coef[2])*ap
        o['H_remainder']=-self.coef[2]*(ep-ap)
        core=o['H_state']+o['H_remainder']
        o['H']=core+self.h_weight[0] if self.history=='fixed' else features@self.h_weight
        o['H_calibration']=o['H']-core
        o['prediction_total']=o['structural_total']+o['H']
        return o
    def save(self,path):
        self.structure.save(path)
        with np.load(path,allow_pickle=False) as d:a={k:d[k].copy() for k in d.files}
        a.update(coef=self.coef,anchor=np.asarray(self.anchor),underfit=np.array(self.underfit),estimator=np.array(self.estimator),history=np.array(self.history),params=np.array(json.dumps(self.params)),mu=np.array(self.mu),h_weight=self.h_weight,start=np.array(self.start))
        np.savez_compressed(path,**a)
    @classmethod
    def load(cls,path):
        s=JointStructural.load(path)
        with np.load(path,allow_pickle=False) as d:
            o=cls(s,d['coef'].copy(),json.loads(str(d['params'])),str(d['estimator']),str(d['history']),tuple(d['anchor']),bool(d['underfit']))
            o.mu=float(d['mu']);o.h_weight=d['h_weight'].copy();o.start=int(d['start'])
        return o
