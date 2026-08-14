# Windows PowerShell run instructions

Unzip the package and enter the folder.

```powershell
py -m venv .venv_eac
.\.venv_eac\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Syntax check:

```powershell
python -m py_compile eac_experiment.py
```

Smoke test only:

```powershell
python eac_experiment.py --preset smoke --output eac_smoke
```

Do not interpret smoke-test p-values.

Final frozen run:

```powershell
python eac_experiment.py --preset confirmatory --output eac_confirmatory_final
```

After completion:

```powershell
Compress-Archive -Path .\eac_confirmatory_final\* -DestinationPath .\eac_confirmatory_final.zip
```

Upload the complete ZIP without editing the CSV files.
