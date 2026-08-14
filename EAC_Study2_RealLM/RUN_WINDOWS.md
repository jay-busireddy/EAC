# Running EAC Study 2 on Windows

## 1. Create a clean environment

```powershell
cd EAC_Study2_RealLM
py -m venv .venv_eac2
.\.venv_eac2\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

If PowerShell blocks activation for the current shell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv_eac2\Scripts\Activate.ps1
```

## 2. Install dependencies

```powershell
python -m pip install torch
python -m pip install -r requirements.txt
```

The first model run downloads the Hugging Face checkpoint and caches it locally. Internet is required for the first download only.

## 3. Syntax check

```powershell
python -m py_compile eac_reallm_study2.py
```

## 4. Software smoke test

This is **not scientific data**. It only verifies model download, LoRA injection, training, evaluation, CSV output, and plots.

```powershell
python eac_reallm_study2.py --preset smoke --output eac2_smoke
```

Expected output folder includes:

- `metrics.csv`
- `hypothesis_tests.csv`
- `run_config.json`
- `SUMMARY.txt`
- `plots\*.png`

Delete or ignore smoke results afterward.

## 5. Optional exploratory pilot

Use this only to estimate runtime and catch software problems. Do not use pilot p-values in the paper.

```powershell
python eac_reallm_study2.py --preset pilot --output eac2_pilot
```

## 6. Frozen confirmatory run

Do not alter parameters after viewing results.

```powershell
python eac_reallm_study2.py --preset confirmatory --output eac2_confirmatory_final
```

For a CPU laptop you may optionally cap PyTorch threads without changing the scientific design:

```powershell
python eac_reallm_study2.py --preset confirmatory --torch-threads 8 --output eac2_confirmatory_final
```

Thread count affects runtime, not the experimental task/hypotheses.

## 7. Zip the results

```powershell
Compress-Archive -Path .\eac2_confirmatory_final\* -DestinationPath .\eac2_confirmatory_final.zip
```

Send `eac2_confirmatory_final.zip` back for independent auditing and manuscript integration.

## 8. Optional larger-model replication

Do this **only after** the primary 135M confirmatory run is complete and frozen.

```powershell
python eac_reallm_study2.py --preset replication360 --output eac2_replication360
Compress-Archive -Path .\eac2_replication360\* -DestinationPath .\eac2_replication360.zip
```

This is a separate replication and should not be used to tune or replace an unfavorable primary result.

## Troubleshooting

### Out of memory
Close other applications. Do not change the confirmatory model or batch size after examining results. If the confirmatory run cannot physically execute on the machine, report that before inspecting partial outcomes; we can define a new protocol rather than silently changing it.

### Model download fails
Retry after confirming Hugging Face access. The checkpoint is public and should not require a token.

### No accepted validated examples
This is a scientifically meaningful outcome, not necessarily a software error. Check `acceptance_rate` in `metrics_partial.csv`. Do not increase candidate samples after inspecting confirmatory data.

### Interrupted run
The script writes `metrics_partial.csv` after each completed seed. Do not merge an interrupted partially tuned run into the confirmatory analysis. If the interruption is purely mechanical, rerun the same frozen preset from scratch into a new empty folder.
