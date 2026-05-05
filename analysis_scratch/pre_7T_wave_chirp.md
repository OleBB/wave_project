# Pre-7T-start wave-chirp characterization (2026-05-05)

## Question

Before claiming the snap-shift Δt under fullwind reflects 'earlier
paddle-wave arrival', we need to confirm that the snap is anchored
in the steady-state paddle-frequency wave train, NOT in the
wavemaker's pre-paddle chirp (long-period → paddle-period sweep).

## Method

Per (freq, amp, wind) cell on canon, take one canonical run.
At IN (9373/170): detect zero upcrossings of η, compute
instantaneous period T_i between consecutive upcrossings. Define
'settle time' as the first upcrossing where |T_i − T_pad|/T_pad < 0.10
for 3 consecutive cycles.

Compare:
- t_exp_7T  — theoretical 7T-start (identical fw/nw)
- t_snap    — where the pipeline actually anchored (per-run)
- t_settle  — first cycle where T_i is within ±10% of T_pad

## Verdict per cell — Δ(snap − settle)

POSITIVE = snap anchored AFTER chirp settled to paddle period (good)
NEGATIVE = snap anchored DURING chirp (bad — measuring non-paddle T)

wind        full    no
f_hz amp_V            
1.3  0.1    4.14  5.83
     0.2    5.00  6.62
     0.3    6.56  6.56
1.4  0.1    3.99  6.14
     0.2    6.07  6.08
     0.3    6.14  6.82
1.5  0.1    3.57  5.77
     0.2    5.60  5.71
     0.3    6.36  6.98
1.6  0.1    4.06  5.39
     0.2    6.54  5.96
     0.3    6.60   NaN

## Implication for the highway-effect Δt

If Δ(snap − settle) is consistently POSITIVE for both fw and nw,
the snap is comfortably in the steady-state wave train and the
earlier-snap-under-fullwind observation is a real timing shift of
the steady-state wave field — the highway effect stands.

If Δ(snap − settle) is NEGATIVE under fullwind but POSITIVE under
nowind (or vice versa), the apparent Δt may be partly due to the
snap landing on different parts of the chirp in the two conditions.
Further investigation needed.

## Files
- pre_7T_wave_chirp_eta.png — η(t) grid 4×3 cells
- pre_7T_wave_chirp_period.png — T_i(t) chirp grid 4×3 cells
- pre_7T_wave_chirp_detail_1.3Hz_0.2V.png — detailed 3×2 panel
- pre_7T_wave_chirp_settle.csv — per-cell settle table
