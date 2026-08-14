#!/usr/bin/env python3
"""
EAC Study 3: competence-gated real-LM adaptation + learned thought scheduling
+ remembering-capacity/path experiments.

This study is intentionally a NEW experiment. It does not alter or rerun Study 2.

Part A (real LM):
  - frozen pretrained causal LM + LoRA
  - pre-EAC external exposure continues only until a prespecified adequacy gate
    is met or a fixed maximum exposure budget is exhausted
  - one-attempt pass/fail verifier (no corrected label is ever injected)
  - compares replay, unchecked self-training, deterministic hard selection,
    uniform selection, learned/function-approximated selection, recombination,
    and anchor replay

Part B (memory/path):
  - controls memory capacity and retention policy
  - evaluates useful path availability under FIXED retrieval bandwidth
  - adds an endogenous rehearsal phase on a disjoint training-query set
  - tests whether larger effective memory increases retrievable paths and whether
    EAC benefits depend on retained structure

Primary hypotheses are frozen in STUDY3_PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from huggingface_hub import model_info
    HAVE_LM = True
except Exception:
    AutoTokenizer = AutoModelForCausalLM = LoraConfig = get_peft_model = None
    model_info = None
    HAVE_LM = False


REAL_CONDITIONS = [
    "no_update",
    "replay",
    "unchecked_hard",
    "validated_uniform",
    "validated_hard",
    "validated_learned",
    "validated_recombine",
    "validated_hard_anchor",
]

MEM_CONDITIONS = [
    "small_selective",
    "large_selective",
    "large_random",
    "small_selective_eac",
    "large_selective_eac",
]


@dataclass
class Config:
    # General
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    real_seeds: int = 12
    real_seed_start: int = 52001
    real_seed_stride: int = 113
    memory_seeds: int = 60
    memory_seed_start: int = 73001
    memory_seed_stride: int = 37

    # Real-LM task sizes
    zorp_external: int = 72
    zorp_cal: int = 32
    zorp_adapt: int = 64
    zorp_seen_eval: int = 24
    zorp_recombine_pool: int = 32
    zorp_recombine_eval: int = 32
    mira_external: int = 80
    mira_cal: int = 32
    mira_eval: int = 96

    # Competence gate
    adequacy_acc: float = 0.75
    adequacy_margin_over_majority: float = 0.15
    adequacy_min_pass_seeds: int = 10
    external_epoch_block: int = 2
    external_max_epochs: int = 18

    # LoRA / optimization
    batch_size: int = 16
    external_lr: float = 5e-4
    adapt_lr: float = 3e-4
    adapt_steps: int = 20
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    max_length: int = 64
    candidate_budget: int = 40
    learned_warmup: int = 12
    scheduler_hidden: int = 12
    scheduler_steps: int = 160
    scheduler_lr: float = 0.03
    anchor_fraction: float = 0.25

    # Memory/path study
    memory_nodes: int = 120
    memory_backbone_edges: int = 180
    memory_distractor_edges: int = 220
    memory_small_capacity: int = 75
    memory_large_capacity: int = 190
    memory_query_train: int = 40
    memory_query_eval_pos: int = 60
    memory_query_eval_neg: int = 60
    memory_max_depth: int = 4
    memory_branch_budget: int = 2
    memory_thought_budget: int = 120
    memory_reinforce_eta: float = 0.20


def preset(name: str) -> Config:
    if name == "smoke":
        return Config(
            real_seeds=1, real_seed_start=99001, real_seed_stride=1,
            memory_seeds=4, memory_seed_start=99101, memory_seed_stride=7,
            zorp_external=40, zorp_cal=16, zorp_adapt=16,
            zorp_seen_eval=16, zorp_recombine_pool=12, zorp_recombine_eval=12,
            mira_external=40, mira_cal=16, mira_eval=32,
            external_max_epochs=2, external_epoch_block=1,
            adapt_steps=2, candidate_budget=8, learned_warmup=4,
            scheduler_steps=20, batch_size=8,
            memory_nodes=60, memory_backbone_edges=70, memory_distractor_edges=80,
            memory_small_capacity=30, memory_large_capacity=75,
            memory_query_train=10, memory_query_eval_pos=12, memory_query_eval_neg=12,
            memory_thought_budget=20,
            adequacy_min_pass_seeds=1,
        )
    if name == "confirmatory":
        return Config()
    raise ValueError(name)


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


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

def paired_test(a, b, alternative="greater"):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    d = a - b
    n = len(d)
    if n == 0:
        return dict(n=0, mean_diff=np.nan, ci_low=np.nan, ci_high=np.nan,
                    t=np.nan, p_one_sided=np.nan, dz=np.nan,
                    wilcoxon_p=np.nan, wins=0, ties=0, losses=0, sign_p=np.nan)
    md = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else np.nan
    if n > 1 and sd > 0:
        se = sd / math.sqrt(n)
        t = md / se
        p = float(stats.t.sf(t, n - 1)) if alternative == "greater" else float(stats.t.cdf(t, n - 1))
        crit = float(stats.t.ppf(.975, n - 1))
        lo, hi = md - crit * se, md + crit * se
        dz = md / sd
    elif n > 1 and sd == 0:
        t = math.inf if md > 0 else (-math.inf if md < 0 else 0.0)
        p = 0.0 if md > 0 else 1.0
        lo = hi = md
        dz = math.inf if md > 0 else (0.0 if md == 0 else -math.inf)
    else:
        t = p = lo = hi = dz = np.nan
    try:
        w = float(stats.wilcoxon(d, alternative=alternative, zero_method="wilcox").pvalue) if np.any(d != 0) else 1.0
    except Exception:
        w = np.nan
    wins, ties, losses = int(np.sum(d > 0)), int(np.sum(d == 0)), int(np.sum(d < 0))
    nz = wins + losses
    signp = float(stats.binomtest(wins, nz, .5, alternative="greater").pvalue) if nz else 1.0
    return dict(n=n, mean_diff=md, ci_low=lo, ci_high=hi, t=t,
                p_one_sided=p, dz=dz, wilcoxon_p=w,
                wins=wins, ties=ties, losses=losses, sign_p=signp)


def holm(ps):
    ps = np.asarray(ps, float)
    out = np.full(len(ps), np.nan)
    finite = np.where(np.isfinite(ps))[0]
    if len(finite) == 0:
        return out
    f = ps[finite]
    m = len(f)
    order = np.argsort(f)
    running = 0.0
    adj = np.empty(m)
    for rank, idx in enumerate(order):
        val = (m - rank) * f[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    out[finite] = adj
    return out


# -----------------------------------------------------------------------------
# Part A: Real-LM EAC
# -----------------------------------------------------------------------------

def choose_binary_labels(tok):
    candidate_sets = [[" A", " B"], ["A", "B"], [" 0", " 1"], ["0", "1"]]
    for labels in candidate_sets:
        ids = [tok.encode(x, add_special_tokens=False) for x in labels]
        if all(len(z) == 1 for z in ids) and len({z[0] for z in ids}) == 2:
            return labels, [z[0] for z in ids]
    raise RuntimeError("Could not find two distinct single-token labels.")


def lm_prompt(task: str, a: int, b: int) -> str:
    return (
        f"Hidden-rule task {task}. The observed pair is a={a}, b={b}. "
        "Predict the class. Answer with exactly one class symbol:"
    )


def make_binary_rule(rng: np.random.Generator, kind: str):
    # Study 3 deliberately uses easier, exactly balanced binary rule families so
    # the source model can first demonstrate real competence before EAC is tested.
    # The seed-specific label flip still requires external exposure.
    flip = int(rng.integers(0, 2))
    return {"kind": kind, "flip": flip}


def binary_true(rule, a: int, b: int) -> int:
    if rule["kind"] == "xor_half":
        # Whether the two symbols come from the same half of the symbol space.
        raw = int((a >= 8) == (b >= 8))
    elif rule["kind"] == "first_half":
        # A second, distinct and balanced task used for retention.
        raw = int(a >= 8)
    else:
        raise ValueError(rule["kind"])
    return raw ^ int(rule["flip"])


def binary_verify(rule, a: int, b: int, proposed: int) -> bool:
    return int(proposed) == binary_true(rule, a, b)


def stratified_take(pool, rule, n, rng):
    by = {0: [], 1: []}
    for p in pool:
        by[binary_true(rule, p[0], p[1])].append(p)
    for k in by:
        rng.shuffle(by[k])
    out = []
    target0 = n // 2
    target1 = n - target0
    for k, need in [(0, target0), (1, target1)]:
        take = min(need, len(by[k]))
        out.extend(by[k][:take])
        by[k] = by[k][take:]
    # Fill any shortage from either class.
    leftovers = by[0] + by[1]
    rng.shuffle(leftovers)
    if len(out) < n:
        out.extend(leftovers[:n-len(out)])
    rng.shuffle(out)
    remaining = [p for p in pool if p not in set(out)]
    return out, remaining


def make_real_splits(seed: int, cfg: Config, zrule, mrule):
    rng = np.random.default_rng(seed + 17)
    all_pairs = [(a,b) for a in range(16) for b in range(16)]

    # Hold out a balanced set of PAIR COMPOSITIONS, not an entire numeric quadrant.
    # Each operand value remains available in external exposure through other pairings.
    # This avoids the class-imbalance/degeneracy seen when a whole quadrant is withheld.
    for attempt in range(100):
        shuffled = all_pairs.copy(); rng.shuffle(shuffled)
        withheld, seen = stratified_take(shuffled, zrule,
                                         cfg.zorp_recombine_pool + cfg.zorp_recombine_eval, rng)
        z_ext, rem = stratified_take(seen, zrule, cfg.zorp_external, rng)
        avals={a for a,b in z_ext}; bvals={b for a,b in z_ext}
        if len(avals)==16 and len(bvals)==16:
            break
    else:
        raise RuntimeError("Could not construct externally exposed split with full operand coverage.")

    z_cal, rem = stratified_take(rem, zrule, cfg.zorp_cal, rng)
    z_adapt, rem = stratified_take(rem, zrule, cfg.zorp_adapt, rng)
    z_seen_eval, rem = stratified_take(rem, zrule, cfg.zorp_seen_eval, rng)
    z_recomb_pool, whrem = stratified_take(withheld, zrule, cfg.zorp_recombine_pool, rng)
    z_recomb_eval, whrem = stratified_take(whrem, zrule, cfg.zorp_recombine_eval, rng)

    mall = all_pairs.copy(); rng.shuffle(mall)
    m_ext, remm = stratified_take(mall, mrule, cfg.mira_external, rng)
    m_cal, remm = stratified_take(remm, mrule, cfg.mira_cal, rng)
    m_eval, remm = stratified_take(remm, mrule, cfg.mira_eval, rng)
    return z_ext, z_cal, z_adapt, z_seen_eval, z_recomb_pool, z_recomb_eval, m_ext, m_cal, m_eval

def labeled(task, pairs, rule):
    return [(task, a, b, binary_true(rule, a, b)) for a,b in pairs]


def unlabeled(task, pairs):
    return [(task, a, b) for a,b in pairs]


def detect_lora_targets(model):
    suffixes = {name.split(".")[-1] for name, _ in model.named_modules()}
    pref = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    targets = [x for x in pref if x in suffixes]
    if not targets:
        raise RuntimeError("Could not locate standard LoRA projection modules.")
    return targets


def capture_lora(model):
    return {k: v.detach().cpu().clone() for k,v in model.state_dict().items() if "lora_" in k}


def restore_lora(model, state):
    sd = model.state_dict()
    with torch.no_grad():
        for k,v in state.items():
            sd[k].copy_(v.to(sd[k].device, dtype=sd[k].dtype))


def lora_norm(model):
    vals = []
    for n,p in model.named_parameters():
        if "lora_" in n:
            vals.append(float(torch.sum(p.detach().float()**2).cpu()))
    return float(math.sqrt(sum(vals))) if vals else 0.0


def encode_prompts(tok, rows, device, max_length):
    texts = [lm_prompt(r[0], r[1], r[2]) for r in rows]
    enc = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {k:v.to(device) for k,v in enc.items()}


def class_probs(model, tok, label_ids, rows, device, cfg, batch=32):
    model.eval(); outs=[]
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            part = rows[i:i+batch]
            enc = encode_prompts(tok, part, device, cfg.max_length)
            out = model(**enc)
            last = enc["attention_mask"].sum(dim=1) - 1
            logits = out.logits[torch.arange(len(part), device=device), last]
            cls = logits[:, torch.tensor(label_ids, device=device)]
            outs.append(torch.softmax(cls.float(), dim=-1).cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.zeros((0,2))


def train_steps(model, tok, label_ids, rows, device, cfg, steps, lr, rng_seed):
    if not rows or steps <= 0:
        return 0
    seed_all(rng_seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    order = np.arange(len(rows)); cursor=0
    model.train()
    for _ in range(steps):
        if cursor == 0:
            np.random.shuffle(order)
        idx=[]
        while len(idx) < min(cfg.batch_size, len(rows)):
            take = min(cfg.batch_size-len(idx), len(order)-cursor)
            idx.extend(order[cursor:cursor+take].tolist())
            cursor += take
            if cursor >= len(order):
                cursor=0; np.random.shuffle(order)
        part = [rows[j] for j in idx]
        enc = encode_prompts(tok, part, device, cfg.max_length)
        y = torch.tensor([r[3] for r in part], dtype=torch.long, device=device)
        out = model(**enc)
        last = enc["attention_mask"].sum(dim=1)-1
        logits = out.logits[torch.arange(len(part), device=device), last]
        cls = logits[:, torch.tensor(label_ids, device=device)]
        loss = F.cross_entropy(cls.float(), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return steps


def train_one_epoch(model, tok, label_ids, rows, device, cfg, lr, seed):
    steps = max(1, math.ceil(len(rows)/cfg.batch_size))
    return train_steps(model,tok,label_ids,rows,device,cfg,steps,lr,seed)


def eval_labeled(model, tok, label_ids, rows, device, cfg):
    if not rows:
        return {"acc":np.nan,"nll":np.nan,"entropy":np.nan,"brier":np.nan}
    p = class_probs(model,tok,label_ids,rows,device,cfg)
    y = np.array([r[3] for r in rows], dtype=int)
    pred = np.argmax(p,axis=1)
    acc = float(np.mean(pred==y))
    py = np.clip(p[np.arange(len(y)),y],1e-12,1)
    nll = float(-np.mean(np.log(py)))
    ent = float(np.mean(-np.sum(np.clip(p,1e-12,1)*np.log(np.clip(p,1e-12,1)),axis=1)))
    one = np.eye(2)[y]
    brier = float(np.mean(np.sum((p-one)**2,axis=1)))
    return {"acc":acc,"nll":nll,"entropy":ent,"brier":brier}


def majority_baseline(rows):
    y=[r[3] for r in rows]
    if not y: return np.nan
    p=sum(y)/len(y)
    return float(max(p,1-p))


def entropy(p):
    q=np.clip(p,1e-12,1)
    return -np.sum(q*np.log(q),axis=1)


def candidate_features(rows, probs):
    ent=entropy(probs)
    mx=probs.max(axis=1)
    margin=np.abs(probs[:,0]-probs[:,1])
    feats=[]
    for i,r in enumerate(rows):
        _,a,b=r
        feats.append([
            probs[i,0], probs[i,1], ent[i], mx[i], margin[i],
            a/15.0, b/15.0, abs(a-b)/15.0,
            float(a>=8), float(b>=8),
        ])
    return np.asarray(feats,dtype=np.float32), ent


class TinyScheduler(nn.Module):
    def __init__(self, din, hidden):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(din,hidden),nn.Tanh(),nn.Linear(hidden,1))
    def forward(self,x): return self.net(x).squeeze(-1)


def fit_scheduler(X,y,cfg,seed):
    seed_all(seed)
    X=torch.tensor(X,dtype=torch.float32)
    y=torch.tensor(y,dtype=torch.float32)
    m=TinyScheduler(X.shape[1],cfg.scheduler_hidden)
    opt=torch.optim.Adam(m.parameters(),lr=cfg.scheduler_lr)
    for _ in range(cfg.scheduler_steps):
        logits=m(X)
        loss=F.binary_cross_entropy_with_logits(logits,y)
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def propose_argmax(probs):
    return np.argmax(probs,axis=1).astype(int)


def verify_candidates(rows, probs, rule):
    proposed=propose_argmax(probs)
    accepted=[]; audit=[]
    ent=entropy(probs)
    for i,(row,pred) in enumerate(zip(rows,proposed)):
        task,a,b=row
        ok=binary_verify(rule,a,b,int(pred))
        audit.append({"task":task,"a":a,"b":b,"proposed_class":int(pred),"verifier_pass":int(ok),"entropy":float(ent[i]),"max_prob":float(probs[i].max())})
        if ok:
            accepted.append((task,a,b,int(pred)))
    return accepted,audit


def select_uniform_pool(model,tok,label_ids,pool,device,cfg,n,rng):
    k=min(n,len(pool)); idx=rng.choice(len(pool),size=k,replace=False)
    rows=[pool[i] for i in idx]
    probs=class_probs(model,tok,label_ids,rows,device,cfg)
    return rows,probs


def select_hard_pool(model,tok,label_ids,pool,device,cfg,n):
    probs=class_probs(model,tok,label_ids,pool,device,cfg)
    ent=entropy(probs); idx=np.argsort(-ent)[:min(n,len(pool))]
    return [pool[i] for i in idx], probs[idx]


def select_learned_pool(model,tok,label_ids,pool,rule,device,cfg,rng,seed):
    # Total verifier calls remain <= candidate_budget. Warmup is part of the budget.
    warm=min(cfg.learned_warmup, cfg.candidate_budget, len(pool))
    idx_w=rng.choice(len(pool),size=warm,replace=False)
    warm_rows=[pool[i] for i in idx_w]
    warm_probs=class_probs(model,tok,label_ids,warm_rows,device,cfg)
    warm_acc,warm_audit=verify_candidates(warm_rows,warm_probs,rule)
    # Fit pass/fail function approximator from candidate features. Ground-truth class is never supplied.
    Xw,_=candidate_features(warm_rows,warm_probs)
    yw=np.array([a["verifier_pass"] for a in warm_audit],dtype=np.float32)
    sched=fit_scheduler(Xw,yw,cfg,seed)

    remain_idx=[i for i in range(len(pool)) if i not in set(idx_w.tolist())]
    remain=[pool[i] for i in remain_idx]
    if not remain or cfg.candidate_budget<=warm:
        return warm_rows,warm_probs,warm_audit,0.0
    rp=class_probs(model,tok,label_ids,remain,device,cfg)
    Xr,er=candidate_features(remain,rp)
    with torch.no_grad():
        pass_prob=torch.sigmoid(sched(torch.tensor(Xr,dtype=torch.float32))).numpy()
    # Expected information yield: predicted verification success * uncertainty.
    utility=pass_prob*er
    k=min(cfg.candidate_budget-warm,len(remain))
    pick=np.argsort(-utility)[:k]
    rows=warm_rows+[remain[i] for i in pick]
    probs=np.concatenate([warm_probs,rp[pick]],axis=0)
    # Audit for warmup is not reused as final audit because all candidates are re-verified consistently below.
    return rows,probs,warm_audit,float(np.mean(utility[pick])) if k else 0.0


def expose_until_adequate(model,tok,label_ids,ext,cal_z,cal_m,device,cfg,seed):
    zbase=majority_baseline(cal_z); mbase=majority_baseline(cal_m)
    total_epochs=0; history=[]; passed=False
    while total_epochs < cfg.external_max_epochs:
        for j in range(cfg.external_epoch_block):
            if total_epochs>=cfg.external_max_epochs: break
            train_one_epoch(model,tok,label_ids,ext,device,cfg,cfg.external_lr,seed+100+total_epochs)
            total_epochs+=1
        ez=eval_labeled(model,tok,label_ids,cal_z,device,cfg)
        em=eval_labeled(model,tok,label_ids,cal_m,device,cfg)
        z_ok=(ez["acc"]>=cfg.adequacy_acc and ez["acc"]>=zbase+cfg.adequacy_margin_over_majority)
        m_ok=(em["acc"]>=cfg.adequacy_acc and em["acc"]>=mbase+cfg.adequacy_margin_over_majority)
        passed=bool(z_ok and m_ok)
        history.append({"epochs":total_epochs,"zorp_cal_acc":ez["acc"],"mira_cal_acc":em["acc"],"zorp_majority":zbase,"mira_majority":mbase,"gate_pass":int(passed)})
        if passed: break
    return passed,total_epochs,history


def real_condition(cond,model,exposed_state,tok,label_ids,pools,rules,device,cfg,seed):
    restore_lora(model,exposed_state)
    rng=np.random.default_rng(seed+1009*(REAL_CONDITIONS.index(cond)+1))
    z_ext,z_pool,z_recombine,m_ext=pools
    audit=[]; generated=[]; scheduler_score=np.nan; t0=time.time()

    if cond=="no_update":
        pass
    elif cond=="replay":
        # Equal optimizer step budget. Candidate-query compute is represented by rescoring old examples.
        q=[z_ext[i%len(z_ext)] for i in range(cfg.candidate_budget)]
        _=class_probs(model,tok,label_ids,q,device,cfg)
        generated=z_ext[:min(len(z_ext),max(1,cfg.candidate_budget))]
        train_steps(model,tok,label_ids,generated,device,cfg,cfg.adapt_steps,cfg.adapt_lr,seed+201)
    elif cond=="unchecked_hard":
        rows,probs=select_hard_pool(model,tok,label_ids,z_pool,device,cfg,cfg.candidate_budget)
        prop=propose_argmax(probs)
        for r,p in zip(rows,prop): generated.append((r[0],r[1],r[2],int(p)))
        train_steps(model,tok,label_ids,generated,device,cfg,cfg.adapt_steps,cfg.adapt_lr,seed+202)
        for r,p in zip(rows,prop):
            audit.append({"task":r[0],"a":r[1],"b":r[2],"proposed_class":int(p),"verifier_pass":np.nan,
                          "audit_correct":int(binary_verify(rules["ZORP"],r[1],r[2],int(p)))})
    elif cond in {"validated_uniform","validated_hard","validated_hard_anchor"}:
        if cond=="validated_uniform":
            rows,probs=select_uniform_pool(model,tok,label_ids,z_pool,device,cfg,cfg.candidate_budget,rng)
        else:
            rows,probs=select_hard_pool(model,tok,label_ids,z_pool,device,cfg,cfg.candidate_budget)
        generated,audit=verify_candidates(rows,probs,rules["ZORP"])
        if cond=="validated_hard_anchor" and generated:
            n_anchor=max(1,int(round(cfg.anchor_fraction*len(generated))))
            keep=max(1,len(generated)-n_anchor)
            anchors=[m_ext[i%len(m_ext)] for i in range(n_anchor)]
            generated=generated[:keep]+anchors
        train_steps(model,tok,label_ids,generated,device,cfg,cfg.adapt_steps,cfg.adapt_lr,seed+203)
    elif cond=="validated_learned":
        rows,probs,_,scheduler_score=select_learned_pool(model,tok,label_ids,z_pool,rules["ZORP"],device,cfg,rng,seed+204)
        generated,audit=verify_candidates(rows,probs,rules["ZORP"])
        train_steps(model,tok,label_ids,generated,device,cfg,cfg.adapt_steps,cfg.adapt_lr,seed+204)
    elif cond=="validated_recombine":
        rows,probs=select_hard_pool(model,tok,label_ids,z_recombine,device,cfg,cfg.candidate_budget)
        generated,audit=verify_candidates(rows,probs,rules["ZORP"])
        train_steps(model,tok,label_ids,generated,device,cfg,cfg.adapt_steps,cfg.adapt_lr,seed+205)
    else:
        raise ValueError(cond)

    valid_audit=[a for a in audit if np.isfinite(a.get("verifier_pass",np.nan))]
    accept=float(np.mean([a["verifier_pass"] for a in valid_audit])) if valid_audit else np.nan
    verified_info=float(np.sum([a.get("entropy",0.0)*a["verifier_pass"] for a in valid_audit])/cfg.candidate_budget) if valid_audit else np.nan
    unchecked_err=np.nan
    if cond=="unchecked_hard" and audit:
        unchecked_err=float(1-np.mean([a["audit_correct"] for a in audit]))
    return {
        "accepted_examples":len(generated),
        "accept_rate":accept,
        "verified_information_yield":verified_info,
        "scheduler_predicted_utility":scheduler_score,
        "unchecked_pseudo_error":unchecked_err,
        "adapter_norm":lora_norm(model),
        "adapt_wall_seconds":time.time()-t0,
        "audit":audit,
    }


def run_real_lm(cfg,outdir,resume=False):
    if not HAVE_LM:
        raise RuntimeError("transformers/peft/huggingface_hub are not installed. Install requirements.txt first.")
    os.makedirs(outdir,exist_ok=True)
    device=device_choice(); print("Real-LM device:",device)
    resolved="main"
    if model_info is not None:
        try: resolved=model_info(cfg.model_name).sha or "main"
        except Exception as e: print("Could not resolve model revision:",e)
    tok=AutoTokenizer.from_pretrained(cfg.model_name,revision=resolved)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    tok.padding_side="right"
    labels,label_ids=choose_binary_labels(tok)
    dtype=torch.float16 if device.type=="cuda" else torch.float32
    model=AutoModelForCausalLM.from_pretrained(cfg.model_name,revision=resolved,torch_dtype=dtype)
    model.to(device)
    targets=detect_lora_targets(model)
    lc=LoraConfig(r=cfg.lora_rank,lora_alpha=cfg.lora_alpha,lora_dropout=cfg.lora_dropout,
                  target_modules=targets,bias="none",task_type="CAUSAL_LM")
    model=get_peft_model(model,lc)
    initial=capture_lora(model)

    partial=os.path.join(outdir,"real_metrics_partial.csv")
    existing=pd.read_csv(partial) if resume and os.path.exists(partial) else pd.DataFrame()
    done=set(existing.seed.unique().tolist()) if not existing.empty else set()
    metrics=[] if existing.empty else existing.to_dict("records")
    adequacy_path=os.path.join(outdir,"adequacy_partial.csv")
    adequacy_rows=pd.read_csv(adequacy_path).to_dict("records") if resume and os.path.exists(adequacy_path) else []
    audit_path=os.path.join(outdir,"candidate_audit_partial.csv")
    audit_rows=pd.read_csv(audit_path).to_dict("records") if resume and os.path.exists(audit_path) else []

    for si in range(cfg.real_seeds):
        seed=cfg.real_seed_start+cfg.real_seed_stride*si
        if seed in done:
            print("Skipping completed real seed",seed); continue
        print(f"\n=== Real seed {seed} ({si+1}/{cfg.real_seeds}) ===")
        seed_all(seed); restore_lora(model,initial)
        rng=np.random.default_rng(seed)
        rules={"ZORP":make_binary_rule(rng,"xor_half"),"MIRA":make_binary_rule(rng,"first_half")}
        sp=make_real_splits(seed,cfg,rules["ZORP"],rules["MIRA"])
        zep,zcp,zap,zsev,zrp,zrev,mep,mcp,mev=sp
        z_ext=labeled("ZORP",zep,rules["ZORP"]); z_cal=labeled("ZORP",zcp,rules["ZORP"])
        z_pool=unlabeled("ZORP",zap); z_seen_eval=labeled("ZORP",zsev,rules["ZORP"])
        z_recombine=unlabeled("ZORP",zrp); z_recombine_eval=labeled("ZORP",zrev,rules["ZORP"])
        m_ext=labeled("MIRA",mep,rules["MIRA"]); m_cal=labeled("MIRA",mcp,rules["MIRA"]); m_eval=labeled("MIRA",mev,rules["MIRA"])
        ext=z_ext+m_ext
        passed,epochs,hist=expose_until_adequate(model,tok,label_ids,ext,z_cal,m_cal,device,cfg,seed)
        for h in hist: adequacy_rows.append({"seed":seed,**h})
        exposed=capture_lora(model)
        pre_seen=eval_labeled(model,tok,label_ids,z_seen_eval,device,cfg)
        pre_comp=eval_labeled(model,tok,label_ids,z_recombine_eval,device,cfg)
        pre_mira=eval_labeled(model,tok,label_ids,m_eval,device,cfg)
        print(f"Gate={passed} epochs={epochs} pre_seen={pre_seen['acc']:.3f} pre_comp={pre_comp['acc']:.3f} pre_mira={pre_mira['acc']:.3f}")

        for cond in REAL_CONDITIONS:
            info=real_condition(cond,model,exposed,tok,label_ids,(z_ext,z_pool,z_recombine,m_ext),rules,device,cfg,seed)
            seen=eval_labeled(model,tok,label_ids,z_seen_eval,device,cfg)
            comp=eval_labeled(model,tok,label_ids,z_recombine_eval,device,cfg)
            mira=eval_labeled(model,tok,label_ids,m_eval,device,cfg)
            overall_rows=z_seen_eval+z_recombine_eval
            overall=eval_labeled(model,tok,label_ids,overall_rows,device,cfg)
            row={
                "seed":seed,"condition":cond,"gate_pass":int(passed),"exposure_epochs":epochs,
                "pre_seen_acc":pre_seen["acc"],"pre_comp_acc":pre_comp["acc"],"pre_mira_acc":pre_mira["acc"],
                "zorp_acc":overall["acc"],"seen_acc":seen["acc"],"composition_acc":comp["acc"],"mira_acc":mira["acc"],
                "zorp_nll":overall["nll"],"zorp_entropy":overall["entropy"],"zorp_brier":overall["brier"],
                "accepted_examples":info["accepted_examples"],"accept_rate":info["accept_rate"],
                "verified_information_yield":info["verified_information_yield"],
                "scheduler_predicted_utility":info["scheduler_predicted_utility"],
                "unchecked_pseudo_error":info["unchecked_pseudo_error"],
                "adapter_norm":info["adapter_norm"],"adapt_wall_seconds":info["adapt_wall_seconds"],
            }
            metrics.append(row)
            for a in info["audit"]:
                audit_rows.append({"seed":seed,"condition":cond,**a})
            print(f"  {cond:24s} all={overall['acc']:.3f} comp={comp['acc']:.3f} mira={mira['acc']:.3f} accepted={info['accepted_examples']}")

        pd.DataFrame(metrics).to_csv(partial,index=False)
        pd.DataFrame(adequacy_rows).to_csv(adequacy_path,index=False)
        if audit_rows: pd.DataFrame(audit_rows).to_csv(audit_path,index=False)
        gc.collect()
        if device.type=="cuda": torch.cuda.empty_cache()

    rdf=pd.DataFrame(metrics).sort_values(["seed","condition"]).reset_index(drop=True)
    rdf.to_csv(os.path.join(outdir,"real_metrics.csv"),index=False)
    adf=pd.DataFrame(adequacy_rows); adf.to_csv(os.path.join(outdir,"adequacy.csv"),index=False)
    if audit_rows: pd.DataFrame(audit_rows).to_csv(os.path.join(outdir,"candidate_audit.csv"),index=False)
    return rdf,adf,{"device":str(device),"model_revision":resolved,"labels":labels,"label_ids":label_ids,"lora_targets":targets}


# -----------------------------------------------------------------------------
# Part B: remembering capacity / path availability
# -----------------------------------------------------------------------------

def generate_memory_world(seed,cfg):
    rng=np.random.default_rng(seed)
    n=cfg.memory_nodes
    # Create a backbone with edges that support many short paths.
    useful=set()
    attempts=0
    while len(useful)<cfg.memory_backbone_edges and attempts<cfg.memory_backbone_edges*20:
        u=int(rng.integers(0,n)); step=int(rng.integers(1,7)); v=(u+step)%n
        if u!=v: useful.add((u,v))
        attempts+=1
    distract=set()
    while len(distract)<cfg.memory_distractor_edges:
        u=int(rng.integers(0,n)); v=int(rng.integers(0,n))
        if u!=v and (u,v) not in useful: distract.add((u,v))
    all_edges=list(useful|distract)
    fixed_noise={e:float(rng.normal(0,0.16)) for e in all_edges}
    salience={e:float(0.52 + (0.13 if e in useful else 0.0) + fixed_noise[e]) for e in all_edges}
    return useful,distract,all_edges,salience,rng


def bfs_path(adj,s,t,max_depth,branch_budget=None):
    if s==t: return [s]
    q=deque([(s,[s])]); seen={s}
    while q:
        u,path=q.popleft()
        if len(path)-1>=max_depth: continue
        neigh=adj.get(u,[])
        if branch_budget is not None: neigh=neigh[:branch_budget]
        for v,*_ in neigh:
            if v==t: return path+[v]
            if v not in seen:
                seen.add(v); q.append((v,path+[v]))
    return None


def make_query_sets(useful,n,cfg,rng):
    # Ground-truth adjacency from useful edges only.
    adj=defaultdict(list)
    for u,v in useful: adj[u].append((v,1.0))
    positives=[]; seen=set(); tries=0
    target=cfg.memory_query_train+cfg.memory_query_eval_pos
    while len(positives)<target and tries<target*500:
        s=int(rng.integers(0,n)); t=int(rng.integers(0,n)); tries+=1
        if s==t or (s,t) in seen: continue
        p=bfs_path(adj,s,t,cfg.memory_max_depth,branch_budget=None)
        if p is not None and 2<=len(p)-1<=cfg.memory_max_depth:
            positives.append((s,t)); seen.add((s,t))
    if len(positives)<target:
        raise RuntimeError("Could not generate enough positive memory queries.")
    rng.shuffle(positives)
    qtrain=positives[:cfg.memory_query_train]
    qpos=positives[cfg.memory_query_train:target]
    negatives=[]; tries=0
    while len(negatives)<cfg.memory_query_eval_neg and tries<cfg.memory_query_eval_neg*1000:
        s=int(rng.integers(0,n)); t=int(rng.integers(0,n)); tries+=1
        if s==t: continue
        if bfs_path(adj,s,t,cfg.memory_max_depth,None) is None:
            negatives.append((s,t))
    if len(negatives)<cfg.memory_query_eval_neg:
        # Fallback: choose pairs whose shortest useful path is beyond max_depth.
        while len(negatives)<cfg.memory_query_eval_neg:
            s=int(rng.integers(0,n)); t=int(rng.integers(0,n))
            if s!=t and bfs_path(adj,s,t,cfg.memory_max_depth,None) is None: negatives.append((s,t))
    return qtrain,qpos,negatives


def weighted_sample_without_replacement(edges,salience,k,rng,selective=True):
    k=min(k,len(edges))
    if not selective:
        idx=rng.choice(len(edges),size=k,replace=False)
        return [edges[i] for i in idx]
    w=np.array([salience[e] for e in edges],float)
    # Positive, soft weighting: retention is not deterministic top-k.
    w=np.exp((w-w.mean())/0.22); w=w/w.sum()
    idx=rng.choice(len(edges),size=k,replace=False,p=w)
    return [edges[i] for i in idx]


def build_retrieval_adj(retained,salience,rehearsal=None):
    rehearsal=rehearsal or {}
    adj=defaultdict(list)
    for e in retained:
        u,v=e
        score=float(salience[e]+rehearsal.get(e,0.0))
        adj[u].append((v,score,e))
    for u in adj:
        adj[u].sort(key=lambda x:x[1],reverse=True)
    return adj


def retrieval_path(adj,s,t,cfg):
    if s==t: return [s],[]
    q=deque([(s,[s],[])]); seen={s}
    while q:
        u,path,edges=q.popleft()
        if len(path)-1>=cfg.memory_max_depth: continue
        for v,score,e in adj.get(u,[])[:cfg.memory_branch_budget]:
            if v==t: return path+[v],edges+[e]
            if v not in seen:
                seen.add(v); q.append((v,path+[v],edges+[e]))
    return None,None


def eval_memory(retained,salience,rehearsal,qpos,qneg,cfg):
    adj=build_retrieval_adj(retained,salience,rehearsal)
    pos_hits=0
    for s,t in qpos:
        p,_=retrieval_path(adj,s,t,cfg)
        pos_hits += int(p is not None)
    neg_correct=0
    for s,t in qneg:
        p,_=retrieval_path(adj,s,t,cfg)
        neg_correct += int(p is None)
    pos_recall=pos_hits/len(qpos)
    neg_spec=neg_correct/len(qneg)
    acc=(pos_hits+neg_correct)/(len(qpos)+len(qneg))
    # Number of unique ordered node pairs reachable under the same bounded retrieval rule.
    reach=0
    for s in range(cfg.memory_nodes):
        for t in range(cfg.memory_nodes):
            if s==t: continue
            p,_=retrieval_path(adj,s,t,cfg)
            if p is not None: reach+=1
    return {"path_recall":float(pos_recall),"negative_specificity":float(neg_spec),"query_accuracy":float(acc),"reachable_pairs":int(reach)}


def endogenous_memory_rehearsal(retained,salience,qtrain,cfg,rng):
    rehearsal=defaultdict(float); successes=0; unique_edges=set()
    # Goal-oriented endogenous traversal over remembered structure; successful
    # internally traversed paths are verified by reaching the training-query target.
    for _ in range(cfg.memory_thought_budget):
        s,t=qtrain[int(rng.integers(0,len(qtrain)))]
        adj=build_retrieval_adj(retained,salience,rehearsal)
        # Stochastic walk restricted to current top retrieval neighborhood.
        u=s; path=[]; visited={u}; ok=False
        for depth in range(cfg.memory_max_depth):
            opts=adj.get(u,[])[:max(cfg.memory_branch_budget,3)]
            if not opts: break
            scores=np.array([x[1] for x in opts],float)
            probs=np.exp((scores-scores.max())/0.25); probs=probs/probs.sum()
            j=int(rng.choice(len(opts),p=probs)); v,_,e=opts[j]
            path.append(e); u=v
            if u==t:
                ok=True; break
            if u in visited: break
            visited.add(u)
        if ok:
            successes+=1
            for e in path:
                rehearsal[e]+=cfg.memory_reinforce_eta
                unique_edges.add(e)
    return rehearsal,{"thought_success_rate":successes/cfg.memory_thought_budget,"reinforced_edges":len(unique_edges)}


def run_memory(cfg,outdir):
    rows=[]
    for si in range(cfg.memory_seeds):
        seed=cfg.memory_seed_start+cfg.memory_seed_stride*si
        useful,distr,edges,sal,rng=generate_memory_world(seed,cfg)
        qtrain,qpos,qneg=make_query_sets(useful,cfg.memory_nodes,cfg,rng)
        for ci,cond in enumerate(MEM_CONDITIONS):
            rr=np.random.default_rng(seed+7001*(ci+1))
            selective = cond != "large_random"
            cap = cfg.memory_small_capacity if cond.startswith("small") else cfg.memory_large_capacity
            retained=weighted_sample_without_replacement(edges,sal,cap,rr,selective=selective)
            before=eval_memory(retained,sal,{},qpos,qneg,cfg)
            rehearsal={}; diag={"thought_success_rate":np.nan,"reinforced_edges":0}
            if cond.endswith("_eac"):
                rehearsal,diag=endogenous_memory_rehearsal(retained,sal,qtrain,cfg,rr)
            after=eval_memory(retained,sal,rehearsal,qpos,qneg,cfg)
            useful_retained=sum(1 for e in retained if e in useful)
            rows.append({
                "seed":seed,"condition":cond,"capacity":cap,"selective":int(selective),
                "useful_edges_retained":useful_retained,"useful_fraction":useful_retained/cap,
                "before_path_recall":before["path_recall"],"path_recall":after["path_recall"],
                "before_query_accuracy":before["query_accuracy"],"query_accuracy":after["query_accuracy"],
                "negative_specificity":after["negative_specificity"],"reachable_pairs":after["reachable_pairs"],
                "eac_path_gain":after["path_recall"]-before["path_recall"],
                "eac_accuracy_gain":after["query_accuracy"]-before["query_accuracy"],
                **diag,
            })
    mdf=pd.DataFrame(rows).sort_values(["seed","condition"]).reset_index(drop=True)
    mdf.to_csv(os.path.join(outdir,"memory_metrics.csv"),index=False)
    return mdf


# -----------------------------------------------------------------------------
# Hypotheses / plots / run
# -----------------------------------------------------------------------------

def make_hypothesis_table(real_df,memory_df,cfg):
    rows=[]
    gate_seeds=[]
    if real_df is not None and not real_df.empty:
        gate_by=real_df.groupby("seed")["gate_pass"].max()
        gate_seeds=gate_by[gate_by==1].index.tolist()
        r=real_df[real_df.seed.isin(gate_seeds)]
        specs=[
            ("A1_validated_hard_vs_replay","real_lm","zorp_acc","validated_hard","replay"),
            ("A2_validation_vs_unchecked","real_lm","zorp_acc","validated_hard","unchecked_hard"),
            ("A3_learned_scheduler_vs_uniform","real_lm","zorp_acc","validated_learned","validated_uniform"),
            ("A4_recombination_vs_replay","real_lm","composition_acc","validated_recombine","replay"),
            ("A5_anchor_retention","real_lm","mira_acc","validated_hard_anchor","validated_hard"),
            ("A6_endogenous_gain_vs_no_update","real_lm","zorp_acc","validated_hard","no_update"),
        ]
        for name,fam,metric,tr,co in specs:
            piv=r.pivot(index="seed",columns="condition",values=metric)
            if tr in piv and co in piv:
                tst=paired_test(piv[tr].values,piv[co].values,"greater")
                rows.append({"hypothesis":name,"family":fam,"metric":metric,"treatment":tr,"control":co,
                             "treatment_mean":float(piv[tr].mean()),"control_mean":float(piv[co].mean()),
                             "gate_pass_seeds":len(gate_seeds),**tst})
    if memory_df is not None and not memory_df.empty:
        specs=[
            ("B1_large_vs_small_capacity","memory","path_recall","large_selective","small_selective"),
            ("B2_large_vs_small_capacity_with_EAC","memory","path_recall","large_selective_eac","small_selective_eac"),
            ("B3_selective_vs_random_large_memory","memory","query_accuracy","large_selective","large_random"),
            ("B4_capacity_enables_more_successful_thought_paths","memory","thought_success_rate","large_selective_eac","small_selective_eac"),
        ]
        for name,fam,metric,tr,co in specs:
            piv=memory_df.pivot(index="seed",columns="condition",values=metric)
            tst=paired_test(piv[tr].values,piv[co].values,"greater")
            rows.append({"hypothesis":name,"family":fam,"metric":metric,"treatment":tr,"control":co,
                         "treatment_mean":float(piv[tr].mean()),"control_mean":float(piv[co].mean()),
                         "gate_pass_seeds":np.nan,**tst})
    t=pd.DataFrame(rows)
    if t.empty: return t
    t["family_holm_p"]=np.nan
    for fam,g in t.groupby("family"):
        t.loc[g.index,"family_holm_p"]=holm(g.p_one_sided.values)
    # Global correction across all prespecified primaries. Non-interpretable real-LM tests still
    # receive a numerical p if enough gate-passed seeds exist; gate rule controls decision.
    t["global_holm_p"]=holm(t.p_one_sided.values)
    real_gate_ok=len(gate_seeds)>=cfg.adequacy_min_pass_seeds
    decisions=[]
    for _,r in t.iterrows():
        if r.family=="real_lm" and not real_gate_ok:
            decisions.append("NOT_INTERPRETABLE_ADEQUACY_GATE")
        elif np.isfinite(r.global_holm_p) and r.global_holm_p<0.05:
            decisions.append("REJECT_H0")
        else:
            decisions.append("FAIL_TO_REJECT_H0")
    t["decision"]=decisions
    return t


def make_plots(real_df,memory_df,tests,adequacy_df,outdir,cfg):
    pdir=os.path.join(outdir,"plots"); os.makedirs(pdir,exist_ok=True)
    if real_df is not None and not real_df.empty:
        # Adequacy gate final calibration accuracy per seed
        if adequacy_df is not None and not adequacy_df.empty:
            last=adequacy_df.sort_values(["seed","epochs"]).groupby("seed").tail(1).sort_values("seed")
            fig,ax=plt.subplots(figsize=(10,5))
            x=np.arange(len(last))
            ax.plot(x,last.zorp_cal_acc,marker="o",label="ZORP calibration")
            ax.plot(x,last.mira_cal_acc,marker="o",label="MIRA calibration")
            ax.axhline(cfg.adequacy_acc,linewidth=1,label="Adequacy accuracy threshold")
            ax.set_xlabel("Matched seed index"); ax.set_ylabel("Calibration accuracy"); ax.set_ylim(0,1)
            ax.set_title("Study 3A source-competence gate"); ax.legend()
            fig.tight_layout(); fig.savefig(os.path.join(pdir,"01_adequacy_gate.png"),dpi=180); plt.close(fig)

        g=real_df[real_df.gate_pass==1]
        if not g.empty:
            means=g.groupby("condition")[["zorp_acc","composition_acc","mira_acc"]].mean().reindex(REAL_CONDITIONS)
            x=np.arange(len(means)); w=.25
            fig,ax=plt.subplots(figsize=(12,6))
            ax.bar(x-w,means.zorp_acc,w,label="ZORP overall")
            ax.bar(x,means.composition_acc,w,label="Recombination eval")
            ax.bar(x+w,means.mira_acc,w,label="MIRA retention")
            ax.set_xticks(x); ax.set_xticklabels(means.index,rotation=35,ha="right"); ax.set_ylim(0,1)
            ax.set_ylabel("Accuracy"); ax.set_title("Study 3A real-LM condition means (gate-passed seeds)"); ax.legend()
            fig.tight_layout(); fig.savefig(os.path.join(pdir,"02_real_lm_condition_means.png"),dpi=180); plt.close(fig)

            sched=g[g.condition.isin(["validated_uniform","validated_hard","validated_learned"])]
            sm=sched.groupby("condition")[["accept_rate","verified_information_yield","zorp_acc"]].mean()
            fig,ax=plt.subplots(figsize=(8,5))
            ax.bar(np.arange(len(sm)),sm.verified_information_yield.values)
            ax.set_xticks(np.arange(len(sm))); ax.set_xticklabels(sm.index,rotation=25,ha="right")
            ax.set_ylabel("Verified information yield / verifier call")
            ax.set_title("Thought-selection policies")
            fig.tight_layout(); fig.savefig(os.path.join(pdir,"03_scheduler_yield.png"),dpi=180); plt.close(fig)

    if memory_df is not None and not memory_df.empty:
        mm=memory_df.groupby("condition")[["path_recall","query_accuracy","reachable_pairs","useful_fraction"]].mean().reindex(MEM_CONDITIONS)
        fig,ax=plt.subplots(figsize=(10,5))
        x=np.arange(len(mm)); w=.35
        ax.bar(x-w/2,mm.path_recall,w,label="Positive path recall")
        ax.bar(x+w/2,mm.query_accuracy,w,label="Balanced query accuracy")
        ax.set_xticks(x); ax.set_xticklabels(mm.index,rotation=30,ha="right"); ax.set_ylim(0,1)
        ax.set_ylabel("Rate"); ax.set_title("Study 3B remembering capacity and EAC"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(pdir,"04_memory_capacity_path_accuracy.png"),dpi=180); plt.close(fig)

        fig,ax=plt.subplots(figsize=(8,5))
        e=memory_df[memory_df.condition.isin(["small_selective_eac","large_selective_eac"])]
        for cond in ["small_selective_eac","large_selective_eac"]:
            q=e[e.condition==cond].sort_values("seed")
            ax.plot(np.arange(len(q)),q.eac_path_gain,marker="o",markersize=3,label=cond)
        ax.axhline(0,linewidth=1); ax.set_xlabel("Memory seed index"); ax.set_ylabel("EAC path-recall gain")
        ax.set_title("EAC benefit under different memory capacities"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(pdir,"05_memory_eac_gain.png"),dpi=180); plt.close(fig)

    if tests is not None and not tests.empty:
        y=np.arange(len(tests)); lo=tests.mean_diff-tests.ci_low; hi=tests.ci_high-tests.mean_diff
        fig,ax=plt.subplots(figsize=(11,7))
        ax.errorbar(tests.mean_diff,y,xerr=np.vstack([lo,hi]),fmt="o",capsize=4)
        ax.axvline(0,linewidth=1); ax.set_yticks(y); ax.set_yticklabels(tests.hypothesis)
        ax.set_xlabel("Paired treatment - control"); ax.set_title("Study 3 prespecified primary effects with 95% CIs")
        fig.tight_layout(); fig.savefig(os.path.join(pdir,"06_primary_effects_95ci.png"),dpi=180); plt.close(fig)


def run(cfg,outdir,mode="all",resume=False):
    os.makedirs(outdir,exist_ok=True)
    real_df=adequacy_df=None; real_meta={}
    memory_df=None
    if mode in {"all","real_lm"}:
        real_df,adequacy_df,real_meta=run_real_lm(cfg,outdir,resume=resume)
    if mode in {"all","memory"}:
        memory_df=run_memory(cfg,outdir)
    tests=make_hypothesis_table(real_df,memory_df,cfg)
    tests.to_csv(os.path.join(outdir,"hypothesis_tests.csv"),index=False)
    make_plots(real_df,memory_df,tests,adequacy_df,outdir,cfg)

    config=asdict(cfg)
    config.update({"mode":mode,"python":sys.version,"torch":torch.__version__,"real_conditions":REAL_CONDITIONS,
                   "memory_conditions":MEM_CONDITIONS,"primary_hypotheses":10,**real_meta})
    with open(os.path.join(outdir,"run_config.json"),"w") as f: json.dump(config,f,indent=2)

    gate_pass=0
    if real_df is not None and not real_df.empty:
        gate_pass=int(real_df.groupby("seed")["gate_pass"].max().sum())
    with open(os.path.join(outdir,"SUMMARY.txt"),"w") as f:
        f.write("EAC Study 3: competence-gated real-LM + function-approximated thought selection + remembering capacity\n\n")
        if real_df is not None:
            f.write(f"Real-LM adequacy gate: {gate_pass}/{cfg.real_seeds} seeds passed; minimum for confirmatory interpretation = {cfg.adequacy_min_pass_seeds}.\n")
            f.write("Verifier uses exactly ONE model proposal per candidate and returns pass/fail only; no corrected label enters EAC.\n\n")
        f.write(tests.to_string(index=False) if not tests.empty else "No hypothesis table generated.")
        f.write("\n\nInterpretation guardrails:\n")
        f.write("- Real-LM primary claims are non-interpretable if the prespecified source-competence gate fails.\n")
        f.write("- Function approximation is tested as a thought-selection policy, not as proof that cognition must be neural or stochastic.\n")
        f.write("- Memory capacity is evaluated under fixed retrieval bandwidth; raw storage size is not equated with intelligence.\n")
        f.write("- Verifier pass/fail is evaluative information; do not claim literally zero new information.\n")
    print("\nPrimary tests:\n",tests.to_string(index=False) if not tests.empty else "none")
    print("Wrote Study 3 results to",outdir)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--preset",choices=["smoke","confirmatory"],default="smoke")
    ap.add_argument("--mode",choices=["all","real_lm","memory"],default="all")
    ap.add_argument("--output",default="eac_study3_smoke")
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--model",default=None,help="Software-compatibility override only before confirmatory results are inspected.")
    args=ap.parse_args()
    cfg=preset(args.preset)
    if args.model: cfg.model_name=args.model
    run(cfg,args.output,args.mode,args.resume)

if __name__=="__main__":
    main()
