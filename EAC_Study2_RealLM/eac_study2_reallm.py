#!/usr/bin/env python3
"""
EAC Study 2: Real pretrained LM + LoRA + programmatic verifier.

Purpose
-------
Tests whether a frozen pretrained causal language model can improve during an
endogenous adaptation interval in which it receives no new environmental
observations or corrective labels. Candidate problems are constructed from the
already-exposed symbol domain. The model proposes class labels; a programmatic
verifier returns only pass/fail. Accepted self-generated labels may be
consolidated into LoRA adapters.

Primary comparisons are prespecified in STUDY2_PROTOCOL.md.

Important distinction
---------------------
The verifier supplies evaluative information (pass/fail). Therefore this study
supports "no new environmental observations/corrective labels during the EAC
interval", NOT "no new information of any kind".
"""

from __future__ import annotations
import argparse, copy, gc, json, math, os, random, time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
try:
    from huggingface_hub import model_info
except Exception:
    model_info = None


CONDITIONS = [
    "no_update",
    "replay",
    "unchecked_hard",
    "validated_uniform",
    "validated_hard",
    "validated_dream",
    "validated_hard_anchor",
]


@dataclass
class Config:
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    seeds: int = 12
    seed_start: int = 41001
    seed_stride: int = 97
    external_zorp: int = 64
    external_mira: int = 64
    within_adapt: int = 32
    cross_adapt: int = 64
    within_eval: int = 32
    cross_eval: int = 64
    mira_eval: int = 96
    external_epochs: int = 5
    adapt_steps: int = 28
    candidate_budget: int = 48
    attempts_per_candidate: int = 4
    batch_size: int = 16
    external_lr: float = 5e-4
    adapt_lr: float = 3e-4
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    anchor_fraction: float = 0.25
    sampling_temperature: float = 1.0
    max_length: int = 64
    bootstrap_samples: int = 10000


def preset(name: str) -> Config:
    if name == "smoke":
        return Config(
            seeds=1, seed_start=91001, seed_stride=1,
            external_zorp=24, external_mira=24,
            within_adapt=12, cross_adapt=16,
            within_eval=12, cross_eval=16, mira_eval=24,
            external_epochs=1, adapt_steps=3, candidate_budget=8,
            attempts_per_candidate=2, batch_size=8,
            bootstrap_samples=1000,
        )
    if name == "confirmatory":
        return Config()
    raise ValueError(f"unknown preset: {name}")


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_choice() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_labels(tok):
    candidate_sets = [
        [" A", " B", " C", " D"],
        ["A", "B", "C", "D"],
        [" 0", " 1", " 2", " 3"],
        ["0", "1", "2", "3"],
    ]
    for labels in candidate_sets:
        ids = [tok.encode(x, add_special_tokens=False) for x in labels]
        if all(len(z) == 1 for z in ids) and len({z[0] for z in ids}) == 4:
            return labels, [z[0] for z in ids]
    raise RuntimeError("Could not find four distinct single-token labels in tokenizer.")


def prompt(task: str, a: int, b: int) -> str:
    return (
        f"Hidden-rule task {task}. Two observed symbols are a={a} and b={b}. "
        "Predict the class. Answer with exactly one class symbol:"
    )


def make_rule(rng: np.random.Generator):
    ca = int(rng.choice([1, 3]))
    cb = int(rng.choice([1, 2, 3]))
    offset = int(rng.integers(0, 4))
    perm = rng.permutation(4).astype(int).tolist()
    return {"ca": ca, "cb": cb, "offset": offset, "perm": perm}


def true_class(rule, a: int, b: int) -> int:
    raw = (rule["ca"] * a + rule["cb"] * b + rule["offset"]) % 4
    return int(rule["perm"][raw])


def verify(rule, a: int, b: int, proposed: int) -> bool:
    # Intentionally returns only pass/fail. It never supplies a corrected label.
    return int(proposed) == true_class(rule, a, b)


def split_pairs(seed: int, cfg: Config):
    rng = np.random.default_rng(seed)
    lo, hi = list(range(8)), list(range(8, 16))
    same = [(a, b) for a in lo for b in lo] + [(a, b) for a in hi for b in hi]
    cross = [(a, b) for a in lo for b in hi] + [(a, b) for a in hi for b in lo]

    # Guarantee that every operand value 0..15 occurs at least once during
    # external ZORP exposure, while still withholding all cross-context pairs.
    coverage=[]
    for group in (lo,hi):
        for pos,a in enumerate(group):
            coverage.append((a,group[(pos+1)%len(group)]))
    remaining=[p for p in same if p not in set(coverage)]
    rng.shuffle(remaining); rng.shuffle(cross)

    need_same = cfg.external_zorp + cfg.within_adapt + cfg.within_eval
    need_cross = cfg.cross_adapt + cfg.cross_eval
    if need_same > len(same) or need_cross > len(cross) or cfg.external_zorp < len(coverage):
        raise ValueError("Requested split exceeds pair pool or cannot cover all operands")

    z_train = coverage + remaining[:cfg.external_zorp-len(coverage)]
    rem_after=remaining[cfg.external_zorp-len(coverage):]
    within_adapt = rem_after[:cfg.within_adapt]
    within_eval = rem_after[cfg.within_adapt:cfg.within_adapt+cfg.within_eval]
    cross_adapt = cross[:cfg.cross_adapt]
    cross_eval = cross[cfg.cross_adapt:need_cross]

    all_pairs = [(a,b) for a in range(16) for b in range(16)]
    rng.shuffle(all_pairs)
    mira_train = all_pairs[:cfg.external_mira]
    mira_eval = all_pairs[cfg.external_mira:cfg.external_mira+cfg.mira_eval]
    return z_train, within_adapt, within_eval, cross_adapt, cross_eval, mira_train, mira_eval


def examples(task: str, pairs: List[Tuple[int,int]], rule):
    return [(task, a, b, true_class(rule, a, b)) for a,b in pairs]

def unlabeled(task: str, pairs: List[Tuple[int,int]]):
    # Candidate pools intentionally contain no ground-truth label field.
    return [(task, a, b) for a,b in pairs]


def detect_lora_targets(model):
    suffixes = {name.split(".")[-1] for name, _ in model.named_modules()}
    preferred = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    targets = [x for x in preferred if x in suffixes]
    if not targets:
        raise RuntimeError("Could not locate standard projection modules for LoRA.")
    return targets


def capture_lora(model):
    return {k: v.detach().cpu().clone() for k,v in model.state_dict().items() if "lora_" in k}


def restore_lora(model, state):
    sd = model.state_dict()
    with torch.no_grad():
        for k,v in state.items():
            sd[k].copy_(v.to(sd[k].device, dtype=sd[k].dtype))


def lora_norm(model) -> float:
    vals=[]
    for n,p in model.named_parameters():
        if "lora_" in n:
            vals.append(float(torch.sum(p.detach().float()**2).cpu()))
    return float(math.sqrt(sum(vals))) if vals else 0.0


def encode_prompts(tok, rows, device, max_length):
    texts=[prompt(t,a,b) for t,a,b,*_ in rows]
    enc=tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {k:v.to(device) for k,v in enc.items()}


def class_probabilities(model, tok, label_ids, rows, device, cfg: Config, batch=32):
    model.eval(); out_probs=[]
    with torch.no_grad():
        for i in range(0,len(rows),batch):
            part=rows[i:i+batch]
            enc=encode_prompts(tok,part,device,cfg.max_length)
            out=model(**enc)
            last=enc["attention_mask"].sum(dim=1)-1
            logits=out.logits[torch.arange(len(part),device=device),last]
            cls=logits[:,torch.tensor(label_ids,device=device)]
            out_probs.append(torch.softmax(cls.float(),dim=-1).cpu().numpy())
    return np.concatenate(out_probs,axis=0) if out_probs else np.zeros((0,4))


def train_steps(model, tok, label_ids, train_rows, device, cfg: Config, steps: int, lr: float, rng_seed: int):
    if not train_rows or steps <= 0:
        return 0
    seed_all(rng_seed)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=lr)
    model.train()
    order=np.arange(len(train_rows)); cursor=0
    completed=0
    for step in range(steps):
        if cursor == 0:
            np.random.shuffle(order)
        idx=[]
        while len(idx)<min(cfg.batch_size,len(train_rows)):
            take=min(cfg.batch_size-len(idx),len(order)-cursor)
            idx.extend(order[cursor:cursor+take].tolist())
            cursor+=take
            if cursor>=len(order):
                cursor=0; np.random.shuffle(order)
        part=[train_rows[j] for j in idx]
        enc=encode_prompts(tok,part,device,cfg.max_length)
        y=torch.tensor([r[3] for r in part],dtype=torch.long,device=device)
        out=model(**enc)
        last=enc["attention_mask"].sum(dim=1)-1
        logits=out.logits[torch.arange(len(part),device=device),last]
        cls=logits[:,torch.tensor(label_ids,device=device)]
        loss=F.cross_entropy(cls.float(),y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); completed+=1
    return completed


def train_epochs(model,tok,label_ids,rows,device,cfg,epochs,lr,seed):
    steps=max(1,math.ceil(len(rows)/cfg.batch_size))*epochs
    return train_steps(model,tok,label_ids,rows,device,cfg,steps,lr,seed)


def eval_rows(model,tok,label_ids,rows,device,cfg):
    if not rows:
        return {"acc":np.nan,"nll":np.nan,"brier":np.nan,"entropy":np.nan}
    probs=class_probabilities(model,tok,label_ids,rows,device,cfg)
    y=np.array([r[3] for r in rows],dtype=int)
    pred=np.argmax(probs,axis=1)
    acc=float(np.mean(pred==y))
    py=np.clip(probs[np.arange(len(y)),y],1e-12,1.0)
    nll=float(-np.mean(np.log(py)))
    one=np.eye(4)[y]
    brier=float(np.mean(np.sum((probs-one)**2,axis=1)))
    ent=float(np.mean(-np.sum(np.clip(probs,1e-12,1)*np.log(np.clip(probs,1e-12,1)),axis=1)))
    return {"acc":acc,"nll":nll,"brier":brier,"entropy":ent}


def entropy_of(probs):
    p=np.clip(probs,1e-12,1.0)
    return -np.sum(p*np.log(p),axis=1)


def select_hard(model,tok,label_ids,pool,device,cfg,n):
    probs=class_probabilities(model,tok,label_ids,pool,device,cfg)
    ent=entropy_of(probs)
    order=np.argsort(-ent)
    k=min(n,len(pool))
    return [pool[i] for i in order[:k]], probs[order[:k]], ent[order[:k]]


def select_uniform(model,tok,label_ids,pool,device,cfg,n,rng):
    k=min(n,len(pool))
    idx=rng.choice(len(pool),size=k,replace=False)
    rows=[pool[i] for i in idx]
    probs=class_probabilities(model,tok,label_ids,rows,device,cfg)
    ent=entropy_of(probs)
    return rows,probs,ent


def sample_from_probs(p, rng, temperature=1.0):
    p=np.asarray(p,dtype=float)
    if temperature != 1.0:
        logp=np.log(np.clip(p,1e-12,1.0))/temperature
        p=np.exp(logp-logp.max()); p=p/p.sum()
    return int(rng.choice(4,p=p/p.sum()))


def verified_self_examples(selected,probs,rule,cfg,rng):
    accepted=[]; attempts=0; correct_first=0
    audit=[]
    for row,p in zip(selected,probs):
        task,a,b=row
        first=None; passed=False; proposed=None
        for j in range(cfg.attempts_per_candidate):
            proposed=sample_from_probs(p,rng,cfg.sampling_temperature)
            attempts+=1
            if first is None: first=proposed
            ok=verify(rule,a,b,proposed)
            if j==0 and ok: correct_first+=1
            audit.append((task,a,b,j,proposed,int(ok)))
            if ok:
                # Crucially: store the model-proposed label that passed. The verifier
                # returns only pass/fail and never inserts a corrected target.
                accepted.append((task,a,b,proposed))
                passed=True
                break
        # candidate may be discarded if every proposal fails
    return accepted,attempts,correct_first,audit


def unchecked_self_examples(selected,probs,rule,cfg,rng):
    rows=[]; err=[]
    for row,p in zip(selected,probs):
        task,a,b=row
        proposed=sample_from_probs(p,rng,cfg.sampling_temperature)
        rows.append((task,a,b,proposed))
        # Ground truth is consulted only after proposal for audit reporting. It is
        # never supplied to this condition's optimizer.
        err.append(int(not verify(rule,a,b,proposed)))
    return rows,float(np.mean(err)) if err else np.nan


def condition_run(cond, model, exposed_state, tok, label_ids, pools, rules, device, cfg, seed):
    restore_lora(model,exposed_state)
    rng=np.random.default_rng(seed + 1009*(CONDITIONS.index(cond)+1))
    z_train, within_pool, cross_pool, mira_train = pools
    candidate_pool=within_pool+cross_pool
    generated=[]; audit=[]; acceptance=np.nan; firstpass=np.nan; pseudo_err=np.nan
    t0=time.time()

    if cond == "no_update":
        pass
    elif cond == "replay":
        # Spend the same candidate-query budget by re-scoring replay examples,
        # then perform the same number of adapter update steps on external labels.
        replay_candidates=[z_train[i%len(z_train)] for i in range(cfg.candidate_budget)]
        _=class_probabilities(model,tok,label_ids,replay_candidates,device,cfg)
        generated=[z_train[i%len(z_train)] for i in range(max(1,min(len(z_train),cfg.candidate_budget)))]
        train_steps(model,tok,label_ids,generated,device,cfg,cfg.adapt_steps,cfg.adapt_lr,seed+101)
    elif cond == "unchecked_hard":
        selected,probs,_=select_hard(model,tok,label_ids,candidate_pool,device,cfg,cfg.candidate_budget)
        generated,pseudo_err=unchecked_self_examples(selected,probs,rules["ZORP"],cfg,rng)
        train_steps(model,tok,label_ids,generated,device,cfg,cfg.adapt_steps,cfg.adapt_lr,seed+102)
    elif cond in {"validated_uniform","validated_hard","validated_hard_anchor"}:
        if cond == "validated_uniform":
            selected,probs,_=select_uniform(model,tok,label_ids,candidate_pool,device,cfg,cfg.candidate_budget,rng)
        else:
            selected,probs,_=select_hard(model,tok,label_ids,candidate_pool,device,cfg,cfg.candidate_budget)
        generated,attempts,first,audit=verified_self_examples(selected,probs,rules["ZORP"],cfg,rng)
        acceptance=len(generated)/len(selected) if selected else 0.0
        firstpass=first/len(selected) if selected else 0.0
        if cond == "validated_hard_anchor" and generated:
            n_anchor=max(1,int(round(cfg.anchor_fraction*len(generated))))
            anchors=[mira_train[i%len(mira_train)] for i in range(n_anchor)]
            # Keep total update-example pool size approximately fixed by replacing,
            # not adding, target examples with anchors.
            keep=max(1,len(generated)-n_anchor)
            generated=generated[:keep]+anchors
        train_steps(model,tok,label_ids,generated,device,cfg,cfg.adapt_steps,cfg.adapt_lr,seed+103)
    elif cond == "validated_dream":
        # Recombination condition: operands were individually observed in same-context
        # examples, but these low/high and high/low pairs were never externally shown.
        selected,probs,_=select_hard(model,tok,label_ids,cross_pool,device,cfg,cfg.candidate_budget)
        generated,attempts,first,audit=verified_self_examples(selected,probs,rules["ZORP"],cfg,rng)
        acceptance=len(generated)/len(selected) if selected else 0.0
        firstpass=first/len(selected) if selected else 0.0
        train_steps(model,tok,label_ids,generated,device,cfg,cfg.adapt_steps,cfg.adapt_lr,seed+104)
    else:
        raise ValueError(cond)

    return {
        "accepted_examples": len(generated),
        "accept_rate": acceptance,
        "first_pass_rate": firstpass,
        "unchecked_pseudo_error": pseudo_err,
        "adapter_norm": lora_norm(model),
        "adapt_wall_seconds": time.time()-t0,
        "audit": audit,
    }


def paired_test(a,b,alternative="greater"):
    a=np.asarray(a,float); b=np.asarray(b,float); d=a-b; n=len(d)
    md=float(d.mean()); sd=float(d.std(ddof=1)) if n>1 else float("nan")
    if n>1 and sd>0:
        se=sd/math.sqrt(n); t=md/se; p=float(stats.t.sf(t,n-1)) if alternative=="greater" else float(stats.t.cdf(t,n-1))
        crit=float(stats.t.ppf(.975,n-1)); lo=md-crit*se; hi=md+crit*se; dz=md/sd
    elif n>1 and sd==0:
        t=math.inf if md>0 else (-math.inf if md<0 else 0.0); p=0.0 if md>0 else 1.0; lo=hi=md; dz=math.inf if md>0 else 0.0
    else:
        t=p=lo=hi=dz=float("nan")
    try:
        w=stats.wilcoxon(d,alternative=alternative,zero_method="wilcox").pvalue if np.any(d!=0) else 1.0
    except Exception:
        w=float("nan")
    wins=int(np.sum(d>0)); ties=int(np.sum(d==0)); losses=int(np.sum(d<0))
    nz=wins+losses
    signp=float(stats.binomtest(wins,nz,.5,alternative="greater").pvalue) if nz else 1.0
    return dict(mean_diff=md,ci_low=lo,ci_high=hi,t=t,p_one_sided=p,dz=dz,wilcoxon_p=float(w),wins=wins,ties=ties,losses=losses,sign_p=signp)


def holm(ps):
    ps=np.asarray(ps,float); m=len(ps); order=np.argsort(ps); adj=np.empty(m,float); running=0.0
    for rank,idx in enumerate(order):
        val=(m-rank)*ps[idx]; running=max(running,val); adj[idx]=min(1.0,running)
    return adj


def hypothesis_table(df):
    # treatment, control, metric, direction. All are treatment > control.
    specs=[
        ("R1_validated_hard_vs_replay","validated_hard","replay","zorp_acc"),
        ("R2_validation_vs_unchecked","validated_hard","unchecked_hard","zorp_acc"),
        ("R3_endogenous_gain_vs_no_update","validated_hard","no_update","zorp_acc"),
        ("R4_dream_recombination_vs_replay","validated_dream","replay","cross_acc"),
        ("R5_anchor_retention","validated_hard_anchor","validated_hard","mira_acc"),
        ("R6_hard_selection_vs_uniform","validated_hard","validated_uniform","zorp_acc"),
    ]
    rows=[]
    for name,tr,co,metric in specs:
        piv=df.pivot(index="seed",columns="condition",values=metric)
        tst=paired_test(piv[tr].values,piv[co].values,"greater")
        rows.append({"hypothesis":name,"metric":metric,"treatment":tr,"control":co,
                     "treatment_mean":float(piv[tr].mean()),"control_mean":float(piv[co].mean()),**tst})
    adj=holm([r["p_one_sided"] for r in rows])
    for r,a in zip(rows,adj):
        r["holm_p"]=float(a); r["reject_h0_0.05"]=bool(a<.05)
    return pd.DataFrame(rows)


def plots(df,tests,outdir):
    pdir=os.path.join(outdir,"plots"); os.makedirs(pdir,exist_ok=True)
    means=df.groupby("condition")[["zorp_acc","cross_acc","mira_acc"]].mean().reindex(CONDITIONS)
    x=np.arange(len(means)); width=.25
    fig,ax=plt.subplots(figsize=(12,6))
    ax.bar(x-width,means["zorp_acc"],width,label="ZORP all")
    ax.bar(x,means["cross_acc"],width,label="Cross-context")
    ax.bar(x+width,means["mira_acc"],width,label="MIRA retention")
    ax.set_xticks(x); ax.set_xticklabels(means.index,rotation=35,ha="right")
    ax.set_ylim(0,1); ax.set_ylabel("Accuracy"); ax.set_title("Study 2: Real-LM EAC condition means"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(pdir,"01_condition_accuracy_means.png"),dpi=180); plt.close(fig)

    y=np.arange(len(tests)); lo=tests.mean_diff-tests.ci_low; hi=tests.ci_high-tests.mean_diff
    fig,ax=plt.subplots(figsize=(10,6))
    ax.errorbar(tests.mean_diff,y,xerr=np.vstack([lo,hi]),fmt="o",capsize=4)
    ax.axvline(0,linewidth=1); ax.set_yticks(y); ax.set_yticklabels(tests.hypothesis)
    ax.set_xlabel("Paired treatment - control"); ax.set_title("Prespecified primary effects with 95% CIs")
    fig.tight_layout(); fig.savefig(os.path.join(pdir,"02_primary_effects_95ci.png"),dpi=180); plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5))
    for cond in ["replay","unchecked_hard","validated_uniform","validated_hard","validated_dream"]:
        g=df[df.condition==cond].sort_values("seed")
        ax.plot(np.arange(len(g)),g.zorp_acc,marker="o",label=cond)
    ax.set_xlabel("Matched seed index"); ax.set_ylabel("ZORP held-out accuracy"); ax.set_ylim(0,1)
    ax.set_title("Seedwise held-out target accuracy"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(pdir,"03_seedwise_zorp_accuracy.png"),dpi=180); plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,5))
    g=df[df.condition.isin(["validated_hard","validated_hard_anchor"])]
    for cond in ["validated_hard","validated_hard_anchor"]:
        q=g[g.condition==cond].sort_values("seed")
        ax.plot(np.arange(len(q)),q.mira_acc,marker="o",label=cond)
    ax.set_xlabel("Matched seed index"); ax.set_ylabel("MIRA retention accuracy"); ax.set_ylim(0,1)
    ax.set_title("Anchor replay retention comparison"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(pdir,"04_anchor_retention.png"),dpi=180); plt.close(fig)


def run(cfg: Config, outdir: str, resume=False):
    os.makedirs(outdir,exist_ok=True)
    device=device_choice()
    print(f"Device: {device}")
    print(f"Loading {cfg.model_name} ...")
    resolved_revision="main"
    if model_info is not None:
        try:
            resolved_revision=model_info(cfg.model_name).sha or "main"
        except Exception as e:
            print("Could not resolve Hub commit SHA; using main:",e)
    print("Model revision:",resolved_revision)
    tok=AutoTokenizer.from_pretrained(cfg.model_name,revision=resolved_revision)
    if tok.pad_token_id is None:
        tok.pad_token=tok.eos_token
    tok.padding_side="right"
    labels,label_ids=choose_labels(tok)
    print("Class labels:",labels,"token ids:",label_ids)

    dtype=torch.float32
    if device.type=="cuda": dtype=torch.float16
    model=AutoModelForCausalLM.from_pretrained(cfg.model_name,revision=resolved_revision,torch_dtype=dtype)
    model.to(device)
    targets=detect_lora_targets(model)
    print("LoRA targets:",targets)
    lc=LoraConfig(r=cfg.lora_rank,lora_alpha=cfg.lora_alpha,lora_dropout=cfg.lora_dropout,
                  target_modules=targets,bias="none",task_type="CAUSAL_LM")
    model=get_peft_model(model,lc)
    initial_state=capture_lora(model)

    partial=os.path.join(outdir,"metrics_partial.csv")
    existing=pd.read_csv(partial) if resume and os.path.exists(partial) else pd.DataFrame()
    done=set(existing.seed.unique().tolist()) if not existing.empty else set()
    all_rows=[] if existing.empty else existing.to_dict("records")
    audit_partial=os.path.join(outdir,"candidate_audit_partial.csv")
    if resume and os.path.exists(audit_partial):
        audit_rows=pd.read_csv(audit_partial).to_dict("records")
    else:
        audit_rows=[]

    for si in range(cfg.seeds):
        seed=cfg.seed_start+cfg.seed_stride*si
        if seed in done:
            print(f"Skipping completed seed {seed}"); continue
        print(f"\n=== seed {seed} ({si+1}/{cfg.seeds}) ===")
        seed_all(seed); restore_lora(model,initial_state)
        rng=np.random.default_rng(seed)
        rules={"ZORP":make_rule(rng),"MIRA":make_rule(rng)}
        splits=split_pairs(seed,cfg)
        z_train_p,within_adapt_p,within_eval_p,cross_adapt_p,cross_eval_p,mira_train_p,mira_eval_p=splits
        z_train=examples("ZORP",z_train_p,rules["ZORP"])
        within_pool=unlabeled("ZORP",within_adapt_p)
        cross_pool=unlabeled("ZORP",cross_adapt_p)
        within_eval=examples("ZORP",within_eval_p,rules["ZORP"])
        cross_eval=examples("ZORP",cross_eval_p,rules["ZORP"])
        z_eval=within_eval+cross_eval
        mira_train=examples("MIRA",mira_train_p,rules["MIRA"])
        mira_eval=examples("MIRA",mira_eval_p,rules["MIRA"])

        ext=z_train+mira_train
        print(f"External exposure: {len(ext)} examples")
        train_epochs(model,tok,label_ids,ext,device,cfg,cfg.external_epochs,cfg.external_lr,seed+1)
        exposed_state=capture_lora(model)
        exposed_z=eval_rows(model,tok,label_ids,z_eval,device,cfg)["acc"]
        print(f"Post-exposure ZORP held-out accuracy: {exposed_z:.3f}")

        seed_rows=[]
        for cond in CONDITIONS:
            info=condition_run(cond,model,exposed_state,tok,label_ids,
                               (z_train,within_pool,cross_pool,mira_train),rules,device,cfg,seed)
            z=eval_rows(model,tok,label_ids,z_eval,device,cfg)
            wi=eval_rows(model,tok,label_ids,within_eval,device,cfg)
            cr=eval_rows(model,tok,label_ids,cross_eval,device,cfg)
            mi=eval_rows(model,tok,label_ids,mira_eval,device,cfg)
            row={
                "seed":seed,"condition":cond,
                "zorp_acc":z["acc"],"zorp_nll":z["nll"],"zorp_brier":z["brier"],"zorp_entropy":z["entropy"],
                "within_acc":wi["acc"],"cross_acc":cr["acc"],"mira_acc":mi["acc"],
                "accepted_examples":info["accepted_examples"],"accept_rate":info["accept_rate"],
                "first_pass_rate":info["first_pass_rate"],"unchecked_pseudo_error":info["unchecked_pseudo_error"],
                "adapter_norm":info["adapter_norm"],"adapt_wall_seconds":info["adapt_wall_seconds"],
            }
            seed_rows.append(row)
            for a in info["audit"]:
                audit_rows.append({"seed":seed,"condition":cond,"task":a[0],"a":a[1],"b":a[2],
                                   "attempt":a[3],"proposed_class":a[4],"verifier_pass":a[5]})
            print(f"  {cond:24s} zorp={z['acc']:.3f} cross={cr['acc']:.3f} mira={mi['acc']:.3f} accepted={info['accepted_examples']}")
        all_rows.extend(seed_rows)
        pd.DataFrame(all_rows).to_csv(partial,index=False)
        if audit_rows: pd.DataFrame(audit_rows).to_csv(os.path.join(outdir,"candidate_audit_partial.csv"),index=False)
        gc.collect()
        if device.type=="cuda": torch.cuda.empty_cache()

    df=pd.DataFrame(all_rows).sort_values(["seed","condition"]).reset_index(drop=True)
    df.to_csv(os.path.join(outdir,"metrics.csv"),index=False)
    tests=hypothesis_table(df)
    tests.to_csv(os.path.join(outdir,"hypothesis_tests.csv"),index=False)
    if audit_rows: pd.DataFrame(audit_rows).to_csv(os.path.join(outdir,"candidate_audit.csv"),index=False)
    plots(df,tests,outdir)
    import sys
    config=asdict(cfg); config.update({"device":str(device),"model_revision":resolved_revision,
                                      "python":sys.version,"torch":torch.__version__,
                                      "labels":labels,"label_ids":label_ids,"lora_targets":targets,
                                      "primary_hypotheses":6,"conditions":CONDITIONS})
    with open(os.path.join(outdir,"run_config.json"),"w") as f: json.dump(config,f,indent=2)

    with open(os.path.join(outdir,"SUMMARY.txt"),"w") as f:
        f.write("EAC Study 2: Real pretrained LM + LoRA + programmatic verifier\n")
        f.write(f"Model: {cfg.model_name}\nSeeds: {cfg.seeds}\nDevice: {device}\n\n")
        f.write(tests.to_string(index=False))
        f.write("\n\nInterpretation guardrail: verifier pass/fail is evaluative information. "
                "Do not describe the study as learning with literally zero new information.\n")
    print("\nPrimary tests:\n",tests.to_string(index=False))
    print("\nWrote results to",outdir)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--preset",choices=["smoke","confirmatory"],default="smoke")
    ap.add_argument("--output",default="eac_study2_smoke")
    ap.add_argument("--model",default=None,help="Override model id. Do not change for confirmatory run after inspecting results.")
    ap.add_argument("--resume",action="store_true",help="Resume only fully completed seeds from metrics_partial.csv")
    args=ap.parse_args()
    cfg=preset(args.preset)
    if args.model: cfg.model_name=args.model
    run(cfg,args.output,args.resume)

if __name__=="__main__":
    main()
