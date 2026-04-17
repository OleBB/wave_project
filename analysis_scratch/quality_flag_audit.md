# Quality-flag audit (per-run)

Generated: 2026-04-17T17:36:59Z

Source: `analysis_scratch/quality_flag_audit.py` → this doc

- Total runs: **646**
- Wave runs: **555**  ·  nowave runs: **91**

## quality_flag distribution

| flag | count |
|---|---|
| `ok` | 619 |
| `dropout_critical` | 23 |
| `in_probe_low_snr` | 3 |
| `probe_malfunction_critical` | 1 |

## Layer 1 — Probe malfunction

Layer 0b/0c in processor.py. Detects stuck sample runs and DC steps. Runs where a malfunction segment overlaps the analysis window are flagged. If the IN or OUT probe is affected → `_critical`, else `_secondary`.

Flag values: `probe_malfunction_critical, probe_malfunction_secondary`
Runs flagged: **1**

### Flagged runs

| folder_file | flag | affected_probes | wave_cond | wind |
| --- | --- | --- | --- | --- |
| 20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange/fullpanel-fullwind-amp0100-freq1600-per40-depth580-mstop30-run1-P2Malfunction.csv | probe_malfunction_critical | 18000/250, 9373/250, 12400/170, 12400/340, 8804/170, 11800/250, 12400/250 | 1.6 Hz / 0.1 V | full |

## Layer 2 — Dropout in analysis window

cut_samples_{pos} / window_size > 2% at the IN or OUT probe. Ignored for rows already flagged by Layer 1 (no downgrade).

Threshold: `cut_samples / window_size > 0.02` at IN or OUT probe
Runs flagged as `dropout_critical`: **23**
Runs near-threshold (max_frac in [0.015, 0.02)): **1**

### Flagged runs

| folder_file | max_cut_frac | freq | amp | wind |
| --- | --- | --- | --- | --- |
| 20251113-tett6roof-loosepaneltaped/reversepanel-fullwind-amp0900-freq1300-per22-depth580-mstop30-run1.csv | 0.037 | 1.300 | 0.900 | full |
| 20260305-newProbePos-tett6roof/nopanel-nowind-amp0400-freq1600-per40-depth580-mstop30-run1.csv | 0.037 | 1.600 | 0.400 | no |
| 20260313-ProbePos4_31_FPV_2-tett6roof/fullpanel-nowind-amp0300-freq1600-per240-depth580-mstop30-run1.csv | 0.037 | 1.600 | 0.300 | no |
| 20260313-ProbePos4_31_FPV_2-tett6roof/fullpanel-nowind-amp0600-freq1300-per40-depth580-mstop0-run1.csv | 0.049 | 1.300 | 0.600 | no |
| 20260321-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-RENAMED/fullpanel-nowind-amp0200-freq1500-per240-depth580-mstop30-run1.csv | 1.000 | 1.500 | 0.200 | no |
| 20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100/fullpanel-fullwind-amp0200-freq1500-per40-depth580-mstop30-run1-P2malfunction.csv | 1.000 | 1.500 | 0.200 | full |
| 20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100/fullpanel-fullwind-amp0300-freq1400-per40-depth580-mstop30-run1-P2malfunction.csv | 1.000 | 1.400 | 0.300 | full |
| 20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100/fullpanel-fullwind-amp0300-freq1500-per40-depth580-mstop30-run1-P2malfunction.csv | 1.000 | 1.500 | 0.300 | full |
| 20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100/fullpanel-nowind-amp0200-freq1800-per40-depth580-mstop30-run1.csv | 0.305 | 1.800 | 0.200 | no |
| 20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100/fullpanel-nowave-depth580-mstop30-nestenstille-run1.csv | 1.000 | — | — |  |
| 20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height136/fullpanel-fullwind-amp0300-freq1700-per40-depth580-mstop30-run1.csv | 1.000 | 1.700 | 0.300 | full |
| 20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height136/fullpanel-fullwind-amp0300-freq1600-per40-depth580-mstop30-run1.csv | 1.000 | 1.600 | 0.300 | full |
| 20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height136/fullpanel-fullwind-amp0300-freq1300-per40-depth580-mstop30-run1.csv | 1.000 | 1.300 | 0.300 | full |
| 20260324-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100/fullpanel-fullwind-nowave-depth580-run-endofday-P2malfunction-butfirstpartcanbeused.csv | 0.041 | — | — | full |
| 20260324-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100/fullpanel-fullwind-amp0200-freq1500-per240-depth580-run1-P2malfunction.csv | 1.000 | 1.500 | 0.200 | full |
| 20260324-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100/fullpanel-nowind-amp0300-freq1300-per240-depth580-run1.csv | 0.023 | 1.300 | 0.300 | no |
| 20260325-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100/fullpanel-nowind-amp0200-freq1400-per240-depth580-mstop30-runx-P2malfunction.csv | 1.000 | 1.400 | 0.200 | no |
| 20260325-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100/fullpanel-nowind-amp0300-freq1300-per240-depth580-mstop30.csv | 0.035 | 1.300 | 0.300 | no |
| 20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange/fullpanel-fullwind-amp0300-freq1600-per40-depth580-mstop30-run1-P2Malfunction.csv | 1.000 | 1.600 | 0.300 | full |
| 20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange/experimental-fromZerotoMaxWin-depth580-mstop30-run1-P1malfunctionMidway.csv | 0.591 | — | — |  |
| 20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange/fullpanel-nowind-amp0300-freq1600-per240-depth580-mstop30-run1.csv | 0.154 | 1.600 | 0.300 | no |
| 20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange/fullpanel-nowind-amp0300-freq1600-per40-depth580-mstop30-run1.csv | 0.073 | 1.600 | 0.300 | no |
| 20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange/fullpanel-nowind-amp0300-freq1700-per240-depth580-mstop30-run1-rettetterpå.csv | 0.027 | 1.700 | 0.300 | no |

### Near-threshold (diagnostic — not flagged)

Runs with moderate NaN-dropout that passed the 2% threshold. Worth a look if a trend in the data points at these.

| folder_file | IN_cut_frac | OUT_cut_frac | max_frac |
| --- | --- | --- | --- |
| 20260313-ProbePos4_31_FPV_2-tett6roof/fullpanel-nowind-amp0300-freq1800-per240-depth580-mstop30-run1.csv | 0.016 | 0.000 | 0.016 |

## Layer 3 — IN-probe low SNR (nowind wave runs)

Nowind wave runs with Probe {in_pos} wave_stability < 0.35 → PCHIP-reconstructed signal flattened; FFT amplitude unreliable. Threshold empirical (see processor.py:1607).

Threshold: `Probe {in_pos} wave_stability < 0.35` (nowind wave runs only)
Runs flagged as `in_probe_low_snr`: **3**
Runs near-threshold (ws in [0.35, 0.45)): **5**

### Flagged runs

| folder_file | wave_stability | freq | amp |
| --- | --- | --- | --- |
| 20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100/fullpanel-nowind-amp0300-freq1800-per40-depth580-mstop30-run1.csv | 0.198 | 1.800 | 0.300 |
| 20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange/fullpanel-nowind-amp0300-freq1600-per40-depth580-mstop30-run2.csv | 0.335 | 1.600 | 0.300 |
| 20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange/fullpanel-nowind-amp0300-freq1400-per40-depth580-mstop30-run1.csv | 0.323 | 1.400 | 0.300 |

### Near-threshold (diagnostic — not flagged)

Nowind wave runs with IN wave_stability in [0.35, 0.45). These pass the quality check but are the next candidates for scrutiny if a systematic issue is suspected.

| folder_file | wave_stability | in_position |
| --- | --- | --- |
| 20260312-ProbPos4_31_FPV_2-tett6roof/fullpanel-nowind-amp0300-freq2000-per40-depth580-mstop30-run1.csv | 0.394 | 9373/170 |
| 20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100/fullpanel-nowind-amp0300-freq1600-per40-depth580-mstop30-run1.csv | 0.367 | 9373/170 |
| 20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100/fullpanel-nowind-amp0300-freq1600-per80-depth580-mstop30-run1.csv | 0.413 | 9373/170 |
| 20260324-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100/fullpanel-nowind-amp0300-freq1600-per240-depth580-run1.csv | 0.379 | 9373/170 |
| 20260325-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100/fullpanel-nowind-amp0300-freq1400-per240-depth580-mstop30-run2.csv | 0.428 | 9373/170 |

## Summary

- 27/646 runs have a non-'ok' quality_flag
- Breakdown above; each flag was applied by exactly one layer (layers are evaluated in order; later layers do not downgrade)
- Flagged runs CSV: `analysis_scratch/quality_flag_audit_flagged.csv`

### What the default filter excludes

By default, `apply_experimental_filters(combined_meta, ...)` keeps:
- `ok`
- `probe_malfunction_secondary` (IN/OUT both fine, auxiliary probe broken)

And excludes:
- `probe_malfunction_critical`
- `dropout_critical`
- `in_probe_low_snr`

Override: pass `filters['quality_flag'] = 'all'` in plotvariables to include everything.

### Re-generate

```bash
python analysis_scratch/quality_flag_audit.py
```
