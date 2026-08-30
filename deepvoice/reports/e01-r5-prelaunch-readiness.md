EXPERIMENT_PRELAUNCH: READY_NOT_LAUNCHED

# E01-R5 visible-CMD full-training readiness

- R4 authorizing audit: PASS, SHA-256 `1aac0423b4e655e349bcbdb47f4cbce0a7df24eeabcf9c4638905778b9ea9203`
- R4↔R5 common source: 15/15 byte-identical
- staging tests: 22/22 PASS
- actual tests: 22/22 PASS
- actual preflight/cache: READY / PASS, 5,651 entries
- fixed training: guarded FP32, batch 32, workers 2, 32,768 samples/epoch × 20 epochs × 3 seeds
- training started: false
- progress: flushed every 25 training batches and 25 validation files
- resume: atomic checkpoint after every completed epoch; code/config/cache/manifest/R4-audit hashes required
- Ctrl-C: status becomes `INTERRUPTED`; latest completed epoch remains resumable
- completed seed skip: only a strict complete-seed JSON with matching artifact hashes is accepted
- assistant polling: disabled; the user will notify completion

## Visible CMD launch command

```powershell
Start-Process -FilePath 'cmd.exe' -ArgumentList @('/k', 'cd /d "C:\Users\MY PC\Desktop\Hackathon\deepvoice" && ".venv-e01\Scripts\python.exe" -u -m experiments.e01_r5.run_full_training')
```

Expected completion files:

- `reports/e01-r5-full-training-result.json`
- `reports/e01-r5-full-training-report.md`
- `reports/e01-r5-code-inventory.csv`
- ignored live state under `artifacts/e01_r5/`
- ignored model checkpoints under `checkpoints/e01_r5/`
