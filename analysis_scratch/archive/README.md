# Archive — analysis_scratch

Diagnostic and superseded files from the 1.6Hz nowind reconstruction investigation (2026-04-15).

| File | What it was |
|------|-------------|
| `reconstruct_run2_test.py` | First prototype: hard-cap + LSQ sine fill for run2 only. No edge buffer, no derivative-guided fill. Superseded by `reconstruct_all3.py`. |
| `reconstruct_run2_test.png` | Output of the above. Run2 FFT looked roughly right but cliff artifacts present at gap re-entries. |
| `zoom_artifacts.png` | Zoom into t≈27.4s and t≈31.8s showing the cliff discontinuity *before* the derivative-guided fill fix. Evidence of the problem. |
| `zoom_reconstructed.png` | Zoom after adding edge buffer (±4 samples) but *before* derivative-guided fill. Cliff still visible. |
| `cliff_check.png` | Zoom into the same regions *after* derivative-guided fill. Confirms the fix worked. Kept here because the authoritative result is `reconstruct_all3.png`. |

Current authoritative scripts are one level up: `reconstruct_all3.py`, `period_overlay.py`, `reconstruct_variants.py`.
