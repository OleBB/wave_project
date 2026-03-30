# Open Questions and Anomalies
*Autonomous analysis session, 2026-03-28/29*

---

## Q1: Why does wind increase OUT probe amplitude at high frequencies?

The OUT probe is behind the panel, sheltered from wind fetch. Yet at 1.7 Hz:
- OUT amplitude full-wind = 7.71 mm
- OUT amplitude no-wind = 4.25 mm (+82%)

The IN probe also increases under wind (15.5 mm vs 12.1 mm, +28%), but the OUT probe increases **more** proportionally. This drives the wind-increases-transmission result.

Hypotheses:
a) Wind is generating wave energy that diffracts around the panel and reaches the OUT probe
b) Wind-driven panel motion (direct push on panel) is generating wave energy on the far side
c) Wind affects the wave reflection pattern between panel and paddle, changing where the amplitude nodes/antinodes are, and if the IN probe happens to sit near an antinode when wind is absent, it would show artificially high IN amplitude when no wind
d) The mstop (wave-stop) happens differently under wind, and the selected "stable wave" window includes a different part of the ramp
e) There's a nonlinear wave-wind-panel interaction that preferentially enhances shorter waves (higher freq) in transmission

**The most physically interesting possibility**: The panel geometry acts as a wave radiator under wind forcing. Wind pushes the panel, the panel oscillates, and this generates waves on both sides. If the panel's natural frequency matches the paddle frequency, this could be a resonance effect. FPV panels in water can have heave resonance frequencies.

---

## Q2: Why does the mooring metadata column say `above_50` for all runs? — RESOLVED

**Root cause found and fixed (2026-03-28).**

The bug was in `extract_metadata_from_filename` in `improved_data_loader.py` line 493. The function passed `filename = Path(path).name` (the CSV basename only, e.g. `fullpanel-fullwind-amp0100-freq1300...csv`) to `_extract_mooring_condition`. That function looks for keywords like `under9Mooring` in the string — but those keywords are only in the **experiment folder name**, not in the individual CSV filenames.

So `_extract_mooring_condition` always returned False for under9 folders, triggering the date-based fallback `get_mooring_type(file_date)`. But that function has a secondary bug: the `MOORING_CONFIGS` list iteration finds `low_mooring` (valid_from 2025-11-06, valid_until=None) before `below_9_mooring` (valid_from 2026-03-17), so it always returns `above_50` for any date after Nov 6 2025.

**Fix applied**: Changed both `_extract_mooring_condition` calls in `extract_metadata_from_filename` to pass `str(file_path)` instead of `filename`. The full path includes the folder name with the mooring keyword.

**The `get_mooring_type` date overlap bug** (low_mooring has no `valid_until`) is now masked by the folder-keyword detection working correctly. However, if a future run is in a folder WITHOUT the mooring keyword, the date-based fallback will still incorrectly return `above_50` for any run after 2026-03-17. This is a latent bug — the `low_mooring` entry should have `valid_until=datetime(2026, 3, 17)` set.

**Action needed**: After applying the fix, run `main.py --force-recompute` to regenerate the Mooring column in all meta.json caches. Then verify with `combined_meta[combined_meta['file_date'] >= '2026-03-16']['Mooring'].value_counts()`.

---

## Q3: The 1.8 Hz exception to the wind effect

At 1.8 Hz, the wind effect is *negative* (wind gives LOWER OUT/IN = 0.46 vs 0.59 for no-wind). All other frequencies from 1.3–1.9 Hz show positive wind effect. Why?

The data:
- 1.8 Hz full-wind: n=7, mean=0.458, std=0.142
- 1.8 Hz no-wind: n=6, mean=0.586, std=0.440

The no-wind std is very high (0.440). This suggests outliers in the no-wind data. The mean is probably not representative — median would be better here. The no-wind IN amplitude at 1.8 Hz is also much lower (8.2 mm vs 16.0 mm with wind), which is surprising.

**Suspicion**: The no-wind 1.8 Hz runs may have wave range detection issues. At 1.8 Hz with no wind, the wavemaker may be generating less stable waves that the _SNARVEI_CALIB doesn't handle well.

---

## Q4: OUT/IN > 1.0 at low frequencies

At 0.7–0.8 Hz, OUT/IN averages 1.02–1.15 (more amplitude past the panel than before it). This happens under BOTH wind and no-wind conditions:
- 0.7 Hz full-wind: 1.115 (n=5)
- 0.7 Hz no-wind: 1.021 (n=2)
- 0.8 Hz full-wind: 1.154 (n=4)
- 0.8 Hz no-wind: 0.991 (n=5)

Possible explanations:
1. **Standing wave / reflection**: The panel reflects some waves back toward the paddle. The paddle partially reflects these back again. This creates a standing wave between paddle and panel. The IN probe (9373/170) may happen to be near a node at certain frequencies (where the partial standing wave has a minimum), making IN amplitude artificially low. The OUT probe is past the panel and sees only the transmitted wave without the standing wave pattern.
2. **Lateral geometry**: The IN probe is at 170mm from the wall, the OUT probe at 250mm (center). The wave field may not be laterally uniform.
3. **Frequency-dependent focusing**: The panel geometry may focus low-frequency wave energy to the center of the tank where the OUT probe sits.

The no-panel control runs show OUT/IN = 0.98 at 1.3 Hz, suggesting very little spatial variation without the panel. With the panel present at low frequencies, OUT/IN > 1 is physically suspicious.

**Important check**: Look at the `parallel_ratio` column or compare the 9373/340 probe amplitudes to understand lateral variability.

---

## Q5: Single-run outliers contaminate frequency-bin means

Several runs have extreme OUT/IN values that were excluded by the < 1.3 filter:
- `freq1600...mstop30-run1.csv` (20260327): OUT/IN = 70.25 (IN probe near noise floor = 0.15 mm)
- `freq1600...mstop30-run4.csv` (20260327): OUT/IN = 2.37
- `freq1900...mstop30-run1.csv` (20260314): OUT/IN = 1.48

The issue seems to be wave range detection failure — the algorithm selected a stillwater window instead of the stable wave window, giving near-zero IN amplitude. These runs should be flagged in `quality_flags.txt`.

---

## Q6: Are 0.3V and 0.6V amplitude runs comparable to 0.1V and 0.2V?

There are only 100 runs at 0.3V and 1 run at 0.6V (in the main config). The 0.3V runs show slightly higher OUT/IN than 0.1V, suggesting amplitude-dependent transmission. But the 0.3V runs were collected later (March 2026) and may coincide with different mooring configurations. Need to verify that the amplitude effect is observable within the same mooring period.

---

## Q7: Wave stability under wind — is the FFT amplitude reliable?

Under full wind, IN wave_stability drops to 0.42–0.80 depending on frequency. Wave stability near 0.4 means the waveform is very irregular. If the FFT amplitude at the paddle frequency is computed over a window with very irregular waves, the FFT estimate is less reliable.

The question is whether the FFT amplitude is correctly capturing the "paddle wave" amplitude or whether destructive interference within the measurement window is underestimating the true wave amplitude at the IN probe.

If full wind causes the wave packet at the IN probe to be less coherent (interfering with itself over the measurement window), the FFT peak amplitude would be artificially LOW. This would make OUT/IN appear HIGHER than it truly is.

This could explain part of the wind-increases-OUT/IN effect: the IN probe FFT amplitude is reduced by wave incoherence under wind, inflating the apparent OUT/IN ratio.

**Test**: Compare OUT/IN for high-stability vs low-stability full-wind runs within the same frequency bin. If low-stability runs have higher OUT/IN, the artifact hypothesis is supported.

---

## Q8: What's happening at 1.3 Hz — the most-studied frequency?

1.3 Hz has the most data (n=78 quality-filtered runs). Looking at 0.1V runs specifically:
- Full wind: OUT/IN = 0.840 (n=18)
- No wind: OUT/IN = 0.709 (n=30)

This is a 18% difference. With 18 and 30 samples, this should be statistically significant. But the standard deviations are large (full: 0.14, no: 0.07), and many of these runs have DIFFERENT mooring types mixed in.

Breaking down by mooring at 1.3 Hz, 0.1V:
We'd need to separate above-panel vs under-panel moorings. With above-panel mooring ONLY: do we still see wind increasing OUT/IN? This is the critical test.

---

## Q9: Reversed panel — what does it show?

There are 61 standard runs with `PanelCondition='reverse'` (reversed panel orientation), but they're in the OLD probe config (9373/250 → 12400/170). These aren't analyzable with the new probe config data. The reversed-panel effect is therefore not accessible in the current main dataset.

---

## Q10: probe_height_mm is not in the metadata

The `probe_height_mm` variable mentioned in the mission instructions is NOT a column in the metadata. The height information is embedded in the dataset directory name (`height100`, `height136`). Only 6 runs exist with height136 (all at no-wind). Not enough to compare.

---

## Q12: Date-related trends in 1.3 Hz no-wind OUT/IN

Looking at 1.3 Hz no-wind runs over time:
- March 7: 0.687 (n=6)
- March 12: 0.676 (n=2)
- March 13: 0.753 (n=5)
- March 14: 0.705 (n=1)
- March 16: 0.671 (n=1)
- March 19: 0.777 (n=2)
- March 21: 0.722 (n=2)
- March 23: 0.767 (n=5)
- March 25: 0.784 (n=7)
- March 26: 0.772 (n=3)
- March 27: 0.736 (n=11)

There is a slight upward trend from early March (~0.68–0.71) to mid/late March (~0.76–0.78). This could be:
- Mooring change (above → below panel) introduced March 16
- Probe calibration drift
- Water temperature change
- Different experimental protocol

The mooring change (March 16) coincides with the jump from ~0.71 to ~0.77. This is confounding evidence for the mooring effect.

---

## Q16: 20260314 — loose panel and breaking waves during last ~3 high-amp runs

Readme for 20260314 notes:
- "semi-brytende bølger" (semi-breaking waves) at amp0300-freq1900 before the panel
- "et panel har løsnet... det fremste gule panelet. det fikk ligge laust de siste 3 ish kjøringene" (a panel came loose — the front yellow one — and lay loose for the last ~3 runs)

Anomalous OUT/IN values for 20260314 high-amp runs:
- freq1.7 Hz amp0300 fullwind: OUT/IN = **1.784** (extreme outlier)
- freq1.9 Hz amp0300 fullwind: OUT/IN = **1.476** (extreme outlier)
- freq1.8 Hz amp0300 nowind: OUT/IN = **1.242** (borderline)

Normal values at same freq but lower amplitude:
- freq1.7 Hz amp0200: OUT/IN = 0.300 (normal)
- freq1.9 Hz amp0200: OUT/IN = 0.221 (normal)

**Conclusion**: The amp0300 outliers at 1.7–1.9 Hz on 20260314 were recorded with a loose panel AND breaking waves. The OUT/IN values are physically meaningless for the standard damping analysis and must be excluded. The question is identifying WHICH specific runs had the loose panel (the last ~3 runs). Need to compare file timestamps or run numbers to identify them.

**Chronological analysis** (by run start time):
- 14:47 — nowind amp0300-freq1800: OUT/IN = 1.242 (BEFORE panel came loose — different cause, likely detection failure at high-freq no-wind)
- 15:09 — fullwind amp0300-freq1900: OUT/IN = 1.476 (possible loose panel)
- 15:12 — fullwind amp0300-freq1800: OUT/IN = 0.562 (normal!)
- 15:18 — fullwind amp0300-freq1700: OUT/IN = 1.784 (extreme — likely loose panel)
- 15:22 — fullwind amp0300-freq1600: OUT/IN = 0.654 (normal despite being last run)

The 1800 Hz run (15:12) between two outliers is normal — so the "loose panel" explanation alone doesn't hold for all outliers. The 1.7 and 1.9 Hz outliers may have BOTH a loose panel AND a wave detection issue (long per240 run where the detection window is incorrectly placed). The readme also mentions "semi-breaking waves" at 1.9 Hz — breaking at the panel would cause anomalous transmission.

**Recommended action**: Flag amp0300-freq1700 and amp0300-freq1900 on 20260314 fullwind as physically compromised (loose panel + semi-breaking waves). The nowind-freq1800 outlier (1.242) is a separate detection issue unrelated to the panel state.

---

## Q14: 20260321 probe 1 malfunction during amp0200-freq1500

Readme for 20260321 notes: "probe 1 (den skrå) på amp0200-freq1500 har feilet totalt, nå later den som den er på 198 ikke 100 som de andre. resetter ULS-boksen."

In the march2026_better_rearranging configuration, probe 1 is at 8804/250 (the upstream probe). A reading of 198 (mm?) when the actual value should be ~100 suggests a sensor saturation or rail error. The 8804/250 probe is not the IN or OUT probe for damping, but it is used for wave range detection in the 8804 calibration group. A saturated probe would produce a bad waveform and could affect wave range detection.

**Check**: Is there an amp0200-freq1500 run in 20260321 that has anomalous OUT/IN? Also check whether the 8804/250 probe amplitude shows > 100 mm for that run (impossible physically — would confirm the malfunction).

---

## Q15: 20260327 panel submersion at higher amplitude runs — multiple dates affected

The 20260327 readme documents panel submersion at:
- `amp0300-freq1400` (full wind): front panels submerge, then float back up
- `amp0200-freq0800` (implied full wind): panel dives to ~6 cm under
- `amp0300-freq1600`: probe dropouts (run unreliable)
- `amp0200-freq1700`: "et par dropouts" — flagged by author as potentially unreliable
- `amp0300-freq1700`: "masse dropouts" — highly unreliable

This is a broader submersion/dropout problem than just 20260319. The quality filter `OUT/IN < 1.3` may not catch all problematic runs — a submerged panel might give OUT/IN slightly above 1.0 without reaching 1.3.

Also noted: the last `amp01-freq1300-per240-mstop330` run on 20260327 was started too soon after the previous run (insufficient stillwater wait). The author flagged this as needing a check of the initial stillwater period.

**Check**: Extract the OUT/IN values for all amp0200 and amp0300 runs on 20260327, especially at 0.8, 1.3–1.7 Hz under full wind. Compare with no-wind same freq to identify anomalies.

---

## Q13: 20260319 run_category and mooring classification — two bugs to verify

**Background**: Subtask investigation confirmed 20260319 had the panel physically submerged (4 cm under for the front panel). OUT/IN = 1.234 at full wind is explained by this physical anomaly, not a pipeline error.

Two things need checking:

1. **Does 20260319 have its own `run_category` or flag?** The quality_flags column shows "ok" for all runs in the autonomous analysis, which means even the submerged-panel runs pass the current quality filter. A new category like `physical_anomaly` or `panel_submerged` would allow automatic exclusion without raising the OUT/IN threshold arbitrarily. Alternatively, adding a `notes` or `excluded` flag column would work.

2. **Mooring classification bug for 20260319**: The `Mooring` column shows `above_50` for this date. But the folder name is `under9Mooring`. This is the same stale-cache problem as Q2 — but for a date that also has a physical anomaly. After `--force-recompute`, verify that 20260319 gets the correct mooring label. There may be a second issue: what IS the correct mooring for 20260319? If the panel was submerged due to wind setup pushing it down, the mooring type affects whether this can happen. This is a physics-relevant detail for documenting the anomaly.

**Recommended action**: After recompute, check `combined_meta[combined_meta['file_date'] == '20260319'][['Mooring', 'WindCondition', 'PanelCondition', 'OUT/IN (FFT)']]` to see what labels were assigned.

---

## Q11: Why does the "lowest" wind condition barely appear in the data?

From the old probe config, there are `lowest` wind condition runs. But in the main modern config (9373/170 → 12400/250), I see NO `lowest` wind runs — only `full` and `no`. This means we can't do a dose-response analysis (no wind → lowest wind → full wind) with the current probe config. The `lowest` wind runs appear to have been done in November 2025 with the old config only.

**Implication**: Cannot separate "some wind" from "maximum wind" effects. The comparison is binary: full industrial fan speed vs no wind at all.
