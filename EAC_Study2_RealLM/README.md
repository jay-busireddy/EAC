# EAC Study 2 — Real-LM Validation

This package is the next-stage experiment requested after the 40-seed synthetic EAC Study 1.

## Why this study is needed
Study 1 showed that the proposed EAC mechanisms behave as designed, but its strongest limitations were:
- purpose-built synthetic mechanism tests;
- a tiny linear low-rank learner rather than a pretrained transformer;
- oracle-like validators in the LoRA experiments.

Study 2 addresses those concerns by using a real pretrained causal LM and LoRA. The validator is still exact, but it is restricted to **pass/fail** and never provides a corrected target to the endogenous learner. This tests a generate → verify → consolidate loop rather than direct oracle labeling.

## Install
Recommended Python: 3.10–3.12.

```powershell
py -m venv .venv_eac2
.\.venv_eac2\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

If you prefer the CPU-only PyTorch wheel on Windows, install PyTorch first from the official PyTorch instructions, then install the remaining requirements.

## 1. Syntax check
```powershell
python -m py_compile eac_study2_reallm.py
```

## 2. Smoke test
Downloads the model on first use. The smoke seed is not part of the confirmatory seed block.

```powershell
python eac_study2_reallm.py --preset smoke --output eac_study2_smoke
```

Check only that it completes and creates CSVs/plots. Do not tune the study based on smoke performance.

## 3. Confirmatory run
```powershell
python eac_study2_reallm.py --preset confirmatory --output eac_study2_confirmatory
```

If interrupted:
```powershell
python eac_study2_reallm.py --preset confirmatory --output eac_study2_confirmatory --resume
```

This can take substantial time on CPU because it performs repeated LoRA updates for seven matched conditions over 12 seeds. GPU acceleration is used automatically when CUDA is available.

## 4. Package the results
```powershell
Compress-Archive -Path .\eac_study2_confirmatory\* -DestinationPath .\eac_study2_confirmatory.zip
```

Send that ZIP back without rerunning based on whether individual hypotheses look favorable.

## Expected output
- `metrics.csv`
- `hypothesis_tests.csv`
- `candidate_audit.csv`
- `run_config.json`
- `SUMMARY.txt`
- `plots/01_condition_accuracy_means.png`
- `plots/02_primary_effects_95ci.png`
- `plots/03_seedwise_zorp_accuracy.png`
- `plots/04_anchor_retention.png`

## Interpretation boundary
The experiment tests improvement with **no new environmental observations or corrective labels** during endogenous adaptation. The verifier's pass/fail signal is evaluative information. This distinction must remain in the paper.
