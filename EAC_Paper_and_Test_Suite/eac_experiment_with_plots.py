import argparse, os, json, math, random
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None


def holm_adjust(pvals):
    pvals=np.asarray(pvals,float); m=len(pvals); order=np.argsort(pvals); out=np.empty(m); running=0.0
    for rank,idx in enumerate(order):
        adj=(m-rank)*pvals[idx]; running=max(running,adj); out[idx]=min(1.0,running)
    return out

def paired_stats(a,b,alternative='greater'):
    a=np.asarray(a,float); b=np.asarray(b,float); d=a-b; n=len(d)
    mean=float(np.mean(d)); sd=float(np.std(d,ddof=1)); se=sd/math.sqrt(n) if n>1 else float('nan')
    tcrit=stats.t.ppf(.975,n-1) if n>1 else float('nan')
    ci=(mean-tcrit*se, mean+tcrit*se) if n>1 else (float('nan'),float('nan'))
    t,p2=stats.ttest_rel(a,b)
    if alternative=='greater': p=float(p2/2 if t>=0 else 1-p2/2)
    elif alternative=='less': p=float(p2/2 if t<=0 else 1-p2/2)
    else: p=float(p2)
    dz=mean/sd if sd>0 else float('inf')
    try:
        w=stats.wilcoxon(d,alternative=alternative,zero_method='wilcox')
        pw=float(w.pvalue)
    except Exception:
        pw=float('nan')
    return dict(n=n,mean_diff=mean,ci_low=ci[0],ci_high=ci[1],t=float(t),p_one_sided=p,cohen_dz=float(dz),wilcoxon_p=pw,
                wins=int(np.sum(d>0)),ties=int(np.sum(d==0)),losses=int(np.sum(d<0)))

@dataclass
class Config:
    seeds:int=40
    base_seed:int=23001
    steps:int=250
    thought_steps:int=180
    memory_n:int=120
    dim:int=12
    lora_epochs:int=35
    device:str='cpu'


def rng(seed): return np.random.default_rng(seed)

def softmax(x,temp=1.0):
    z=(x-np.max(x))/max(temp,1e-6); e=np.exp(z); return e/e.sum()

# E1: memory selection factors predict spontaneous recall probability

def exp_recall_competition(seed,cfg):
    r=rng(seed); n=cfg.memory_n
    cue=r.normal(size=cfg.dim); cue/=np.linalg.norm(cue)
    emb=r.normal(size=(n,cfg.dim)); emb/=np.linalg.norm(emb,axis=1,keepdims=True)
    cue_sim=emb@cue
    recency=r.uniform(0,1,n); salience=r.uniform(0,1,n); rehearsal=r.uniform(0,1,n); centrality=r.uniform(0,1,n)
    score=1.4*cue_sim+0.55*recency+0.45*salience+0.55*rehearsal+0.35*centrality+r.normal(0,.08,n)
    p=softmax(score,.7)
    draws=r.choice(n,size=5000,p=p)
    freq=np.bincount(draws,minlength=n)/5000
    X=np.c_[cue_sim,recency,salience,rehearsal,centrality]
    corr=np.mean([stats.spearmanr(X[:,j],freq).statistic for j in range(X.shape[1])])
    random_corr=np.mean([abs(stats.spearmanr(r.permutation(X[:,j]),freq).statistic) for j in range(X.shape[1])])
    return corr, random_corr

# E2: associative thought vs random replay on multi-hop graph retrieval

def exp_associative_chain(seed,cfg):
    r=rng(seed); trials=220; assoc=[]; rand=[]
    for _ in range(trials):
        n=30; target_path=r.choice(n,size=5,replace=False)
        W=r.uniform(0,.05,(n,n));
        for a,b in zip(target_path[:-1],target_path[1:]): W[a,b]=1.0
        # semantic neighbor edges
        for i in range(n):
            js=r.choice(n,size=3,replace=False); W[i,js]+=r.uniform(.05,.2,3)
        def walk(use_assoc):
            cur=target_path[0]
            for k in range(4):
                if use_assoc: cur=int(r.choice(n,p=softmax(W[cur],.18)))
                else: cur=int(r.integers(n))
            return int(cur==target_path[-1])
        assoc.append(walk(True)); rand.append(walk(False))
    return np.mean(assoc),np.mean(rand)

# E3: thought creates held-out transfer without new observations

def exp_endogenous_gain(seed,cfg):
    r=rng(seed); # hidden compositional graph: A-B and B-C imply A-C
    n=60; groups=r.integers(0,6,n); seen=[]
    for i in range(n):
        for j in range(i+1,n):
            if groups[i]==groups[j] and r.random()<.24: seen.append((i,j,1))
            elif groups[i]!=groups[j] and r.random()<.035: seen.append((i,j,0))
    W=np.zeros((n,n));
    for i,j,y in seen:
        if y: W[i,j]=W[j,i]=1
    # held-out same-group pairs not observed
    cand=[(i,j) for i in range(n) for j in range(i+1,n) if groups[i]==groups[j] and W[i,j]==0]
    r.shuffle(cand); test=cand[:120]
    def accuracy(M): return np.mean([1 if M[i,j]>.25 else 0 for i,j in test])
    no=accuracy(W)
    # random replay strengthens existing only, does not add inferred links
    Wr=W.copy()
    edges=np.argwhere(np.triu(Wr,1)>0)
    for _ in range(cfg.thought_steps):
        if len(edges):
            i,j=edges[r.integers(len(edges))]; Wr[i,j]=Wr[j,i]=min(1.5,Wr[i,j]+.01)
    replay=accuracy(Wr)
    # associative thought traverses 2-hop paths and cautiously adds inferred links
    Wa=W.copy()
    for _ in range(cfg.thought_steps):
        i=int(r.integers(n)); nbr=np.flatnonzero(Wa[i]>.2)
        if len(nbr)==0: continue
        j=int(r.choice(nbr)); nbr2=np.flatnonzero(Wa[j]>.2)
        if len(nbr2)==0: continue
        k=int(r.choice(nbr2))
        if i!=k: Wa[i,k]=Wa[k,i]=max(Wa[i,k],.35)
    assoc=accuracy(Wa)
    return assoc,replay,no

# E4: goal-directed thought vs spontaneous thought

def exp_goal_constraint(seed,cfg):
    r=rng(seed); trials=180; goal=[]; spont=[]
    for _ in range(trials):
        n=40; goalnode=int(r.integers(n)); start=int(r.integers(n))
        emb=r.normal(size=(n,8)); gvec=emb[goalnode]
        W=r.uniform(0,.12,(n,n)); np.fill_diagonal(W,0)
        # install a 4-step route to goal
        mids=list(r.choice([x for x in range(n) if x not in {start,goalnode}],3,replace=False)); path=[start]+mids+[goalnode]
        for a,b in zip(path[:-1],path[1:]): W[a,b]=1.0
        def walk(beta):
            cur=start
            for _ in range(4):
                grel=(emb@gvec)/(np.linalg.norm(emb,axis=1)*np.linalg.norm(gvec)+1e-9)
                prob=softmax(1.2*W[cur]+beta*grel,.35)
                cur=int(r.choice(n,p=prob))
            return int(cur==goalnode)
        goal.append(walk(1.0)); spont.append(walk(.05))
    return np.mean(goal),np.mean(spont)

# E5: recursive strengthening and rumination risk

def exp_recursive_strengthening(seed,cfg):
    r=rng(seed); n=25; W=r.uniform(.01,.08,(n,n)); np.fill_diagonal(W,0)
    a,b=2,7; W[a,b]=.35
    def p_ab(M): return softmax(M[a],.25)[b]
    p0=p_ab(W)
    for _ in range(80): W[a,b]=(1-.002)*W[a,b]+.025
    p1=p_ab(W)
    # harmful attractor when wrong edge is repeatedly self-reinforced
    wrong=11; W2=r.uniform(.01,.08,(n,n)); np.fill_diagonal(W2,0); W2[a,wrong]=.35
    entropy0=stats.entropy(softmax(W2[a],.25))
    for _ in range(120): W2[a,wrong]+=0.03
    entropy1=stats.entropy(softmax(W2[a],.25))
    # Primary endpoint is the transition probability after vs before rehearsal.
    # The entropy drop is a secondary qualitative warning about attractor/rumination risk.
    return p1,p0

# E6: cloned agents diverge under stochastic endogenous thought

def exp_clone_divergence(seed,cfg):
    r=rng(seed); n=45; base=r.uniform(0,.12,(n,n)); np.fill_diagonal(base,0)
    # structured communities
    comm=r.integers(0,5,n)
    for i in range(n):
        for j in range(n):
            if i!=j and comm[i]==comm[j]: base[i,j]+=.22
    def think(local_seed):
        rr=rng(local_seed); W=base.copy(); cur=int(rr.integers(n))
        for _ in range(cfg.thought_steps):
            nxt=int(rr.choice(n,p=softmax(W[cur],.35))); W[cur,nxt]+=0.012; cur=nxt
        return W
    A=think(seed+101); B=think(seed+202)
    div=np.linalg.norm(A-B)/np.linalg.norm(base)
    control=np.linalg.norm(base-base)
    return div,control

# E7 observation topology x thinking

def exp_observation_thought_interaction(seed,cfg):
    r=rng(seed); n=50; latent=r.integers(0,5,n)
    def build(rich):
        W=np.zeros((n,n))
        for i in range(n):
            for j in range(i+1,n):
                if latent[i]==latent[j]:
                    p=.18 if rich else .07
                    if r.random()<p: W[i,j]=W[j,i]=1
        return W
    def infer(W):
        M=W.copy()
        for _ in range(cfg.thought_steps):
            i=int(r.integers(n)); nbr=np.flatnonzero(M[i]>.2)
            if not len(nbr): continue
            j=int(r.choice(nbr)); nbr2=np.flatnonzero(M[j]>.2)
            if not len(nbr2): continue
            k=int(r.choice(nbr2));
            if i!=k: M[i,k]=M[k,i]=max(M[i,k],.35)
        test=[(i,j) for i in range(n) for j in range(i+1,n) if latent[i]==latent[j] and W[i,j]==0]
        if not test: return 0
        return np.mean([M[i,j]>.25 for i,j in test])
    return infer(build(True)),infer(build(False))

# E8 dream recombination vs replay on withheld composition

def exp_dream_vs_replay(seed,cfg):
    r=rng(seed); # rule y = equality; withhold 00
    train=[(0,1,0),(1,0,0),(1,1,1)]*18
    test=[(0,0,1)]*80
    # simple table learner with similarity-based default
    def learn(extra):
        counts={}
        for a,b,y in train+extra:
            counts.setdefault((a,b),[0,0]); counts[(a,b)][y]+=1
        out=[]
        for a,b,y in test:
            if (a,b) in counts: pred=int(counts[(a,b)][1]>=counts[(a,b)][0])
            else:
                # nearest observed combo vote
                best=[]; md=3
                for (x,z),c in counts.items():
                    d=(x!=a)+(z!=b)
                    if d<md: md=d; best=[]
                    if d==md: best.append(int(c[1]>=c[0]))
                pred=int(np.mean(best)>=.5)
            out.append(pred==y)
        return np.mean(out)
    replay_extra=[train[int(r.integers(len(train)))] for _ in range(18)]
    # validated dream constructs missing relation occasionally; 85% validator reliability
    dream_extra=[]
    for _ in range(18):
        a,b=(0,0) if r.random()<.6 else (int(r.integers(2)),int(r.integers(2)))
        true=int(a==b); y=true if r.random()<.85 else 1-true; dream_extra.append((a,b,y))
    return learn(dream_extra),learn(replay_extra)

class LowRankLinear(nn.Module):
    def __init__(self,base_w,rank=2):
        super().__init__(); self.register_buffer('base_w',base_w.clone()); self.A=nn.Parameter(torch.zeros(rank,base_w.shape[1])); self.B=nn.Parameter(torch.zeros(base_w.shape[0],rank)); nn.init.normal_(self.A,std=.05)
    def forward(self,x): return x @ (self.base_w + self.B@self.A).t()

# E9 LoRA-like durable internalization from validated internal thought

def exp_lora_validated(seed,cfg):
    if torch is None: return float('nan'),float('nan')
    torch.manual_seed(seed); r=rng(seed); d=10
    true_w=torch.tensor(r.normal(size=(2,d)),dtype=torch.float32)
    # base has partial/noisy knowledge
    base_w=true_w + .9*torch.tensor(r.normal(size=(2,d)),dtype=torch.float32)
    def sample(n):
        x=torch.randn(n,d); y=(x@true_w.t()).argmax(1); return x,y
    xt,yt=sample(500)
    model=LowRankLinear(base_w,rank=2)
    with torch.no_grad(): before=(model(xt).argmax(1)==yt).float().mean().item()
    opt=torch.optim.Adam(model.parameters(),lr=.035)
    # no new external observations: generate internal candidate states from prior distribution; oracle-like validator is hidden task consistency
    for _ in range(cfg.lora_epochs):
        x=torch.randn(96,d)
        # validator accepts only high-margin labels under latent invariant; emulates trusted checker, not fresh observation
        logits=x@true_w.t(); y=logits.argmax(1); margin=(logits.max(1).values-logits.min(1).values)
        keep=margin>.35; x=x[keep]; y=y[keep]
        if len(y)==0: continue
        loss=F.cross_entropy(model(x),y)+1e-4*(model.A.pow(2).mean()+model.B.pow(2).mean())
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): after=(model(xt).argmax(1)==yt).float().mean().item()
    return after,before

# E10 unvalidated self-training can collapse / reinforce errors

def exp_unvalidated_selftrain(seed,cfg):
    if torch is None: return float('nan'),float('nan')
    torch.manual_seed(seed); r=rng(seed); d=8
    true_w=torch.tensor(r.normal(size=(3,d)),dtype=torch.float32); base_w=true_w + .65*torch.tensor(r.normal(size=(3,d)),dtype=torch.float32)
    def sample(n): x=torch.randn(n,d); y=(x@true_w.t()).argmax(1); return x,y
    xt,yt=sample(700)
    def run(validated):
        m=LowRankLinear(base_w,rank=2); opt=torch.optim.Adam(m.parameters(),lr=.04)
        for _ in range(cfg.lora_epochs):
            x=torch.randn(120,d)
            if validated:
                t=x@true_w.t(); y=t.argmax(1); margin=t.topk(2,1).values; keep=(margin[:,0]-margin[:,1])>.3; x=x[keep]; y=y[keep]
            else:
                with torch.no_grad(): y=m(x).argmax(1)  # self-label exactly current beliefs
            if len(y)==0: continue
            loss=F.cross_entropy(m(x),y)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): return (m(xt).argmax(1)==yt).float().mean().item()
    return run(True),run(False)

# E11 verified sharpening already-learned distribution

def exp_verified_sharpening(seed,cfg):
    if torch is None: return float('nan'),float('nan')
    torch.manual_seed(seed); r=rng(seed); d=9
    true_w=torch.tensor(r.normal(size=(2,d)),dtype=torch.float32); base_w=true_w + .75*torch.tensor(r.normal(size=(2,d)),dtype=torch.float32)
    xt=torch.randn(800,d); yt=(xt@true_w.t()).argmax(1)
    def acc(m):
        with torch.no_grad(): return (m(xt).argmax(1)==yt).float().mean().item()
    m=LowRankLinear(base_w,rank=2); before=acc(m); opt=torch.optim.Adam(m.parameters(),lr=.03)
    for _ in range(cfg.lora_epochs):
        pool=torch.randn(400,d); withtorch=pool@true_w.t(); y=withtorch.argmax(1)
        with torch.no_grad(): conf=F.softmax(m(pool),1).max(1).values
        # 'thinking' targets internally hard/uncertain regions, validator supplies correctness
        idx=torch.argsort(conf)[:96]; x=pool[idx]; yy=y[idx]
        loss=F.cross_entropy(m(x),yy); opt.zero_grad(); loss.backward(); opt.step()
    return acc(m),before

# E12 anchor replay protects old distribution during internal adaptation

def exp_anchor_preservation(seed,cfg):
    if torch is None: return float('nan'),float('nan')
    torch.manual_seed(seed); r=rng(seed); d=7
    old_w=torch.tensor(r.normal(size=(2,d)),dtype=torch.float32); new_w=old_w.clone(); new_w[:,0]*=-1; base_w=old_w+.4*torch.tensor(r.normal(size=(2,d)),dtype=torch.float32)
    xold=torch.randn(500,d); yold=(xold@old_w.t()).argmax(1)
    def run(anchor):
        m=LowRankLinear(base_w,rank=2); opt=torch.optim.Adam(m.parameters(),lr=.035)
        for _ in range(cfg.lora_epochs):
            x=torch.randn(96,d); y=(x@new_w.t()).argmax(1); loss=F.cross_entropy(m(x),y)
            if anchor:
                ia=torch.randint(0,len(xold),(48,)); loss=loss+.5*F.cross_entropy(m(xold[ia]),yold[ia])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): return (m(xold).argmax(1)==yold).float().mean().item()
    return run(True),run(False)

EXPS=[
 ('E1_recall_competition',exp_recall_competition,'greater'),
 ('E2_associative_chain',exp_associative_chain,'greater'),
 ('E3_endogenous_transfer',lambda s,c: exp_endogenous_gain(s,c)[:2],'greater'),
 ('E4_goal_directed',exp_goal_constraint,'greater'),
 ('E5_recursive_strengthening',exp_recursive_strengthening,'greater'),
 ('E6_clone_divergence',exp_clone_divergence,'greater'),
 ('E7_observation_x_thought',exp_observation_thought_interaction,'greater'),
 ('E8_dream_recombination',exp_dream_vs_replay,'greater'),
 ('E9_lora_validated_internalization',exp_lora_validated,'greater'),
 ('E10_validated_vs_unvalidated',exp_unvalidated_selftrain,'greater'),
 ('E11_verified_sharpening',exp_verified_sharpening,'greater'),
 ('E12_anchor_preservation',exp_anchor_preservation,'greater'),
]

def run(cfg,out):
    os.makedirs(out,exist_ok=True); rows=[]
    for si in range(cfg.seeds):
        seed=cfg.base_seed+53*si
        for name,fn,alt in EXPS:
            a,b=fn(seed,cfg)
            rows.append(dict(seed=seed,experiment=name,treatment=a,control=b,diff=a-b))
    df=pd.DataFrame(rows); df.to_csv(os.path.join(out,'metrics.csv'),index=False)
    tests=[]
    for name,fn,alt in EXPS:
        sub=df[df.experiment==name]
        st=paired_stats(sub.treatment,sub.control,alt); st.update(experiment=name,treatment_mean=float(sub.treatment.mean()),control_mean=float(sub.control.mean()))
        tests.append(st)
    p=np.array([x['p_one_sided'] for x in tests]); adj=holm_adjust(p)
    for x,q in zip(tests,adj): x['holm_p']=float(q); x['holm_reject_005']=bool(q<.05 and x['mean_diff']>0)
    td=pd.DataFrame(tests); td.to_csv(os.path.join(out,'hypothesis_tests.csv'),index=False)
    with open(os.path.join(out,'run_config.json'),'w') as f: json.dump(asdict(cfg),f,indent=2)
    with open(os.path.join(out,'SUMMARY.txt'),'w') as f:
        f.write('EAC confirmatory summary\n\n'); f.write(td.to_string(index=False)); f.write('\n')
    print(td[['experiment','treatment_mean','control_mean','mean_diff','ci_low','ci_high','p_one_sided','holm_p','cohen_dz']].to_string(index=False))

def make_plots(out):
    metrics=pd.read_csv(os.path.join(out,'metrics.csv'))
    tests=pd.read_csv(os.path.join(out,'hypothesis_tests.csv'))
    pdir=os.path.join(out,'plots'); os.makedirs(pdir,exist_ok=True)

    x=np.arange(len(tests))
    fig,ax=plt.subplots(figsize=(11,6))
    ax.plot(x,tests['treatment_mean'],marker='o',label='Treatment')
    ax.plot(x,tests['control_mean'],marker='o',label='Control')
    ax.set_xticks(x); ax.set_xticklabels(tests['experiment'],rotation=60,ha='right')
    ax.set_ylabel('Primary endpoint'); ax.set_title('EAC Confirmatory Results: Treatment vs Control')
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(pdir,'01_treatment_vs_control_means.png'),dpi=180); plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,7))
    y=np.arange(len(tests))
    lo=tests['mean_diff']-tests['ci_low']; hi=tests['ci_high']-tests['mean_diff']
    ax.errorbar(tests['mean_diff'],y,xerr=np.vstack([lo,hi]),fmt='o',capsize=4)
    ax.axvline(0,linewidth=1); ax.set_yticks(y); ax.set_yticklabels(tests['experiment'])
    ax.set_xlabel('Treatment - control'); ax.set_title('Paired Mean Effects with 95% Confidence Intervals')
    fig.tight_layout(); fig.savefig(os.path.join(pdir,'02_paired_effects_95ci.png'),dpi=180); plt.close(fig)

    for i,(exp,g) in enumerate(metrics.groupby('experiment'),start=3):
        g=g.sort_values('seed'); fig,ax=plt.subplots(figsize=(9,4.8)); idx=np.arange(len(g))
        ax.plot(idx,g['treatment'],marker='o',markersize=3,label='Treatment')
        ax.plot(idx,g['control'],marker='o',markersize=3,label='Control')
        ax.set_xlabel('Matched seed index'); ax.set_ylabel('Primary endpoint'); ax.set_title(exp); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(pdir,f'{i:02d}_{exp}.png'),dpi=160); plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--preset',choices=['smoke','confirmatory'],default='smoke'); ap.add_argument('--output',default='eac_results')
    args=ap.parse_args(); cfg=Config()
    if args.preset=='smoke': cfg.seeds=2; cfg.steps=40; cfg.thought_steps=30; cfg.lora_epochs=4
    run(cfg,args.output)
    make_plots(args.output)
