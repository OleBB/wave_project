# Full vs reverse at 1.30 Hz — K_t per (panel, mooring, wind, amp, hardware)

Source: `full_vs_reverse_at_1_3hz_ka_summary.csv`
Plot:   `output/FIGURES/ch05_full_vs_reverse_at_1_3hz_ka.pdf`

Each row = one cluster of points in the figure (one marker shape × one
colour × one fill style). `n` = number of runs in that cluster. `ka` is
the per-run mean of k(1.30 Hz) × IN Amplitude (FFT). `K_t` = OUT/IN (FFT).
`K_t std` is `—` when n=1 (no spread defined).

| panel | mooring | wind | amp | hardware | n | ka (mean) | K_t (mean) | K_t (std) |
|---|---|---|---|---|---:|---:|---:|---:|
| normal | above_50 | med | A1 (0.1V) | earlier | 4 | 0.0551 | 0.6341 | 0.0293 |
| normal | above_50 | med | A2 (0.2V) | earlier | 2 | 0.1056 | 0.7123 | 0.0070 |
| normal | above_50 | med | A3 (0.3V) | earlier | 3 | 0.1622 | 0.7253 | 0.0190 |
| normal | above_50 | uten | A1 (0.1V) | earlier | 9 | 0.0516 | 0.6297 | 0.0381 |
| normal | above_50 | uten | A2 (0.2V) | earlier | 2 | 0.1047 | 0.7228 | 0.0124 |
| normal | above_50 | uten | A3 (0.3V) | earlier | 2 | 0.1548 | 0.7392 | 0.0135 |
| normal | canon | med | A1 (0.1V) | earlier | 5 | 0.0516 | 0.7577 | 0.0318 |
| normal | canon | med | A1 (0.1V) | canon (cond4) | 10 | 0.0510 | 0.7861 | 0.0463 |
| normal | canon | med | A2 (0.2V) | earlier | 3 | 0.1014 | 0.8338 | 0.0227 |
| normal | canon | med | A2 (0.2V) | canon (cond4) | 3 | 0.1008 | 0.8577 | 0.0218 |
| normal | canon | med | A3 (0.3V) | earlier | 1 | 0.1550 | 0.8194 | — |
| normal | canon | med | A3 (0.3V) | canon (cond4) | 4 | 0.1517 | 0.8327 | 0.0173 |
| normal | canon | uten | A1 (0.1V) | earlier | 14 | 0.0515 | 0.6563 | 0.0297 |
| normal | canon | uten | A1 (0.1V) | canon (cond4) | 9 | 0.0515 | 0.6604 | 0.0281 |
| normal | canon | uten | A2 (0.2V) | earlier | 5 | 0.1010 | 0.8000 | 0.0177 |
| normal | canon | uten | A2 (0.2V) | canon (cond4) | 2 | 0.1003 | 0.8106 | 0.0079 |
| normal | canon | uten | A3 (0.3V) | earlier | 1 | 0.1496 | 0.8022 | — |
| normal | canon | uten | A3 (0.3V) | canon (cond4) | 2 | 0.1466 | 0.8297 | 0.0023 |
| revers | above_50 | med | A1 (0.1V) | earlier | 2 | 0.0546 | 0.7055 | 0.0142 |
| revers | above_50 | med | A2 (0.2V) | earlier | 2 | 0.1039 | 0.6854 | 0.0337 |
| revers | above_50 | med | A3 (0.3V) | earlier | 2 | 0.1600 | 0.6844 | 0.0078 |
| revers | above_50 | uten | A1 (0.1V) | earlier | 7 | 0.0516 | 0.6175 | 0.0606 |
| revers | above_50 | uten | A2 (0.2V) | earlier | 5 | 0.1048 | 0.6829 | 0.0350 |
| revers | above_50 | uten | A3 (0.3V) | earlier | 5 | 0.1549 | 0.7136 | 0.0279 |
