# EAC Study 3 — Frozen Confirmatory Protocol

## Purpose
Study 2 produced a valuable negative result: the real LM had not first acquired adequate source competence, and a multi-attempt verifier could behave like rejection sampling over weak guesses. Study 3 is a **new confirmatory study**, not a rerun or rescue of Study 2.

It tests three propositions:

1. **Competence-gated EAC:** endogenous adaptation should be evaluated only after the source model demonstrably learned the externally exposed task.
2. **Function-approximated thinking:** the policy that chooses what to think about may itself be learned rather than deterministic.
3. **Remembering capacity and path availability:** more effective retained structure may permit more useful paths to be retrieved and traversed, but raw storage alone is not equated with intelligence.

Study 2 remains unchanged in the paper regardless of Study 3 outcomes.

---

# Part A — Real pretrained LM, LoRA, competence gate, and learned thought scheduling

## Fixed model
- `HuggingFaceTB/SmolLM2-135M-Instruct`
- frozen pretrained base weights
- PEFT/LoRA on available attention/MLP projection modules
- rank 8, alpha 16, dropout 0

A model change is allowed only for software compatibility discovered during the **smoke** run, before confirmatory outcomes are inspected.

## Tasks
Two balanced binary hidden-rule tasks are used.

### ZORP
The hidden rule depends on whether the two integer symbols are in the same half of the 0--15 symbol domain. A seed-specific output-label flip prevents the base model from knowing the task's label convention without external exposure.

### MIRA
A second balanced hidden rule depends on whether the first symbol is in the high or low half of the symbol domain, again with a seed-specific output-label flip. MIRA serves as an old task for retention testing.

The rule is intentionally easier than Study 2's random four-class modular rule. Study 3 asks what happens **after competence exists**, not whether a 135M LM can infer a difficult arbitrary hidden function from a tiny sample.

## Pair-composition holdout
For ZORP, 64 balanced operand pairs are withheld from external exposure. Each individual operand value still occurs during external exposure through other pairings. The 64 unseen pairings are split into:
- 32 internally available recombination candidates;
- 32 disjoint recombination evaluation pairs.

Thus recombination means producing a never-exposed **pairing of already exposed components**, not introducing a new operand.

## Prespecified source-competence gate
Before the EAC interval, the model receives labeled external ZORP and MIRA examples. Exposure proceeds in fixed two-epoch blocks up to a maximum of 18 epochs.

A seed passes the gate only when **both** independent calibration sets satisfy:

- accuracy >= 0.75, and
- accuracy >= that calibration set's majority-class baseline + 0.15.

The calibration sets are not used for EAC training and are not final evaluation sets.

### Study-level adequacy rule
The six real-LM primary hypotheses are confirmatorily interpretable only if at least **10 of 12** matched seeds pass the gate. If fewer than 10 pass, their numerical results remain diagnostics and the decisions are labeled `NOT_INTERPRETABLE_ADEQUACY_GATE`.

This gate is fixed before the confirmatory run and must not be modified after seeing outcomes.

## Real-LM confirmatory seeds
12 fresh matched seeds:

`52001 + 113*i`, for `i = 0,...,11`.

Smoke seed `99001` is outside the confirmatory block.

## Endogenous interval
After the competence gate/external-exposure phase ends, there are:
- no new environmental observations;
- no corrective target labels supplied to the EAC optimizer.

A verifier may return **one pass/fail bit** for one model-proposed label. Unlike Study 2, there are no repeated attempts on the same candidate. This removes the rejection-sampling artifact in which multiple random guesses can eventually pass.

The verifier never supplies a corrected class.

## Real-LM conditions
All conditions begin from the identical post-exposure LoRA state within each matched seed and use the same LoRA rank and fixed optimizer-step budget.

1. `no_update` — no endogenous adapter update.
2. `replay` — literal reuse of externally labeled ZORP examples.
3. `unchecked_hard` — high-entropy unseen candidates are pseudo-labeled by the model and consolidated without validation.
4. `validated_uniform` — unseen candidates are chosen uniformly and consolidated only when the model's single proposal passes verification.
5. `validated_hard` — high-entropy unseen candidates are selected deterministically and consolidated only when verified.
6. `validated_learned` — a small neural function approximator learns from pass/fail outcomes on a prespecified warmup portion of the verifier budget, then scores remaining candidates by predicted pass probability times uncertainty. This tests whether the **thought-selection function itself can be approximated/learned**.
7. `validated_recombine` — internally constructed never-exposed operand pairings are consolidated only when the model's single proposal passes verification.
8. `validated_hard_anchor` — validated hard EAC with a fixed fraction of old MIRA anchors replacing target examples to test stability/retention.

## Candidate/verifier budgets
- ZORP unseen candidate pool: 64 examples.
- candidate/verifier budget: 40 per validated thinking condition.
- learned scheduler warmup: 12 of the 40 calls; remaining calls are selected by the learned scheduler.
- recombination pool: 32 candidates.
- each candidate receives **exactly one** model proposal and at most one verifier bit.

## Part A primary hypotheses
For every directional hypothesis:

`H0: mu_treatment <= mu_control`

`HA: mu_treatment > mu_control`

### A1 — Validated EAC versus replay
`validated_hard` has higher held-out `zorp_acc` than `replay`.

### A2 — Validation necessity
`validated_hard` has higher held-out `zorp_acc` than `unchecked_hard`.

### A3 — Function-approximated thought selection
`validated_learned` has higher held-out `zorp_acc` than `validated_uniform`.

This does **not** hypothesize that deterministic thought is invalid. It tests whether a learned approximation can add value over unstructured candidate choice under equal verifier budget.

### A4 — Recombination
`validated_recombine` has higher held-out `composition_acc` than `replay` on never-exposed pair compositions.

### A5 — Retention anchors
`validated_hard_anchor` has higher `mira_acc` than `validated_hard`.

### A6 — Endogenous gain
`validated_hard` has higher held-out `zorp_acc` than `no_update`.

## Part A secondary diagnostics
Not additional confirmatory claims:
- calibration trajectories and epochs-to-gate;
- pre-EAC seen and composition accuracy;
- NLL, Brier score, and predictive entropy;
- verifier acceptance rate;
- verified information yield per verifier call;
- learned-scheduler predicted utility;
- unchecked pseudo-label error rate (audit only);
- LoRA adapter norm;
- adaptation wall time;
- learned scheduler versus deterministic hard selection (exploratory).

---

# Part B — Remembering capacity, useful paths, and endogenous traversal

## Motivation
Raw memory capacity is not defined as intelligence. Study 3B asks a narrower question:

> Under a fixed retrieval bandwidth, does retaining more useful episodic structure make more valid relational paths retrievable and more available to endogenous traversal?

This separates at least three variables:
- **capacity** — how much can be retained;
- **retention quality** — which observations survive;
- **retrievability/thought traversal** — which retained paths can actually be activated under a bounded search budget.

## Memory worlds
For each seed, a directed episodic graph contains:
- useful relational edges that generate multi-hop ground-truth paths;
- distractor edges;
- noisy per-edge salience, with useful edges only probabilistically more salient than distractors.

Retention is stochastic rather than deterministic top-k storage.

## Fixed retrieval constraint
All memory conditions use:
- the same maximum traversal depth;
- the same branch/retrieval bandwidth (`top 2` outgoing retained edges at each node).

Therefore larger storage does not automatically imply that every stored edge is searchable. Interference remains possible.

## Memory capacities
- small capacity: 75 retained edges;
- large capacity: 190 retained edges.

## Retention policies
- selective: stochastic retention weighted by noisy salience;
- random: capacity-matched random retention.

## Endogenous traversal
For EAC conditions, the agent receives a fixed budget of internally initiated goal-directed walks over remembered edges. Training-query endpoints are disjoint from evaluation-query endpoints. A traversal is reinforced only when it reaches the internally specified target; no new edges are added.

Thus thought can change **edge accessibility/ranking**, not retroactively create observations that were never retained.

## Memory confirmatory seeds
60 fresh seeds:

`73001 + 37*i`, for `i = 0,...,59`.

Smoke memory seeds are outside this block.

## Part B primary hypotheses

### B1 — Capacity and remembered path availability
`large_selective` has higher held-out positive `path_recall` than `small_selective` under the same retrieval bandwidth.

### B2 — Capacity under endogenous traversal
`large_selective_eac` has higher held-out positive `path_recall` than `small_selective_eac` under the same thought and retrieval budgets.

### B3 — Capacity is not enough; retention quality matters
At the same large capacity, `large_selective` has higher balanced `query_accuracy` than `large_random`.

### B4 — More retained structure permits more successful internally activated paths
Under the same thought budget, `large_selective_eac` has higher `thought_success_rate` than `small_selective_eac`.

## Part B secondary diagnostics
- useful edges retained and useful fraction;
- bounded reachable-pair count;
- positive path recall;
- negative specificity;
- balanced query accuracy;
- pre/post EAC path recall;
- EAC gain;
- number of distinct reinforced edges.

---

# Statistical plan
There are **10 prespecified primary hypotheses** total: A1--A6 and B1--B4.

For each paired comparison:
- one-sided paired t-test;
- mean paired difference;
- two-sided 95% CI for the paired difference;
- paired Cohen `dz`;
- Wilcoxon signed-rank sensitivity test;
- wins/ties/losses;
- one-sided sign test.

The script reports:
- Holm correction within each family (`real_lm`, `memory`);
- a conservative Holm correction across all 10 primaries.

A confirmatory rejection requires **global Holm-adjusted p < .05** and, for Part A, satisfaction of the study-level source-competence gate.

Failure to reject does not prove the null.

# Integrity rules
1. Smoke runs are software checks only.
2. Do not use smoke outcomes as evidence.
3. Do not modify seeds, thresholds, budgets, task rules, endpoints, or hypotheses after inspecting confirmatory results.
4. Run the confirmatory study once.
5. If interrupted, use `--resume` with the same directory; completed real-LM seeds are skipped.
6. Do not rerun failed hypotheses with new settings as part of Study 3. Any revised design becomes Study 4.
7. Return the complete output directory/ZIP, not screenshots alone.
