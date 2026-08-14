# EAC Confirmatory Protocol

## Scope
This suite tests computational claims of Endogenous Associative Cognition (EAC). It does not test consciousness, biological equivalence, or human subjective thought.

## Frozen primary family
Twelve directional hypotheses are tested as one confirmatory family with Holm correction:

1. H1 Recall competition: structured activation factors > shuffled factors.
2. H2 Associative traversal: associative walk > random replay.
3. H3 Endogenous transfer: associative thought > literal replay without new observations.
4. H4 Goal constraint: goal-directed thought > spontaneous thought on goal-constrained tasks.
5. H5 Recursive strengthening: repeated traversal increases transition probability and demonstrates attractor risk.
6. H6 Clone divergence: independent stochastic thought yields greater state divergence than no-thought control.
7. H7 Observation × thought: context-rich observation topology > context-sparse topology after equal thought compute.
8. H8 Dream recombination: validated recombination > equal-budget replay on withheld composition.
9. H9 LoRA durable internalization: validated internal practice improves held-out performance over the unchanged adapter.
10. H10 Validation necessity: validated endogenous adaptation > unvalidated self-label adaptation.
11. H11 Distribution sharpening: verified hard-example practice improves an already partially learned distribution.
12. H12 Preservation: anchor replay during endogenous adaptation preserves prior knowledge better than no-anchor adaptation.

For each A>B claim:

H0: mu_A <= mu_B
HA: mu_A > mu_B

Primary alpha = .05 after Holm family-wise correction.

## Confirmatory seeds
40 matched seeds:

23001 + 53*i, i=0,...,39

Do not add seeds after inspecting results.

## Fixed compute
Paired conditions receive the same latent world, starting representation, and internal-step budget wherever applicable. The LoRA experiments freeze base weights and train only low-rank matrices.

## Outputs
- metrics.csv
- hypothesis_tests.csv
- run_config.json
- SUMMARY.txt

## Interpretation
A significant result supports only the operational mechanism in the synthetic environment. Failure to reject H0 is not proof of no effect. Do not retune coefficients after observing confirmatory p-values. If a hypothesis fails, report it and revise the mechanism rather than extending the run.
