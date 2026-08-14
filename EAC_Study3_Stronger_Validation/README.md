# EAC Study 3 — Run Instructions

Study 3 is a new experiment. **Do not rerun or replace Study 2.**

It contains two parts:

- **Study 3A:** competence-gated real pretrained LM + LoRA + one-bit verifier + deterministic/uniform/learned thought selection.
- **Study 3B:** remembering capacity, selective retention, bounded multi-hop retrieval, and endogenous path traversal.

Read `STUDY3_PROTOCOL.md` before the confirmatory run.

## 1. Create/activate a fresh Windows environment

```powershell
cd EAC_Study3_Stronger_Validation
py -m venv .venv_eac3
.\.venv_eac3\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

If PyTorch installation fails, install the appropriate CPU/CUDA PyTorch build first, then rerun the requirements command.

## 2. Syntax check

```powershell
python -m py_compile eac_study3.py
```

## 3. Smoke test

Run the full software smoke test:

```powershell
python eac_study3.py --preset smoke --mode all --output eac_study3_smoke
```

The smoke test is **not scientific evidence**. Its only purpose is to verify model download, LoRA training, CSV writing, graph-memory execution, and plot generation.

If the real-LM dependency/model load fails, send the error before running confirmatory.

You can separately verify the cheap memory component with:

```powershell
python eac_study3.py --preset smoke --mode memory --output eac_study3_memory_smoke
```

## 4. Frozen confirmatory run

Only after smoke succeeds:

```powershell
python eac_study3.py --preset confirmatory --mode all --output eac_study3_confirmatory
```

If interrupted:

```powershell
python eac_study3.py --preset confirmatory --mode all --output eac_study3_confirmatory --resume
```

Do not change settings after seeing partial confirmatory outcomes.

## 5. Expected outputs

```text
eac_study3_confirmatory/
    real_metrics.csv
    adequacy.csv
    candidate_audit.csv
    memory_metrics.csv
    hypothesis_tests.csv
    run_config.json
    SUMMARY.txt
    plots/
        01_adequacy_gate.png
        02_real_lm_condition_means.png
        03_scheduler_yield.png
        04_memory_capacity_path_accuracy.png
        05_memory_eac_gain.png
        06_primary_effects_95ci.png
```

Partial CSVs may also remain; include them in the ZIP.

## 6. Package results

```powershell
Compress-Archive -Path .\eac_study3_confirmatory\* -DestinationPath .\eac_study3_confirmatory.zip
```

Upload that ZIP back to ChatGPT.

## What will be audited before paper revision

The returned files will be independently checked for:

- exact fresh seed blocks;
- expected condition/seed row counts;
- source-competence gate pass count;
- no multi-attempt verifier leakage;
- candidate acceptance and unchecked pseudo-label error;
- paired primary statistics and confidence intervals;
- family and global Holm correction;
- memory-capacity path coverage and retrieval tradeoffs;
- whether the learned thought scheduler actually adds value;
- whether anchor replay preserves MIRA at a cost to ZORP;
- negative findings and failure modes.

The paper should be updated **after** this audit, not before.
