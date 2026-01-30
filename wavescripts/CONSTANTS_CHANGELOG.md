# Constants Changelog

Track all parameter changes here so you can remember what you tried and why.

## Format
```
Date: YYYY-MM-DD
Changed: CONSTANT_NAME
From: old_value
To: new_value
Reason: Why you changed it
Result: What happened (fill in after testing)
```

---

## 2025-01-30 - Initial values

### Signal Processing
- `BASELINE_SIGMA_FACTOR = 1.0`
  - Reason: Starting point from original code
  - Result: [To be filled in]

- `SMOOTHING_FULL_WIND = 15`
  - Reason: Empirically determined for noisy full-wind conditions
  - Result: Works well for 1.3 Hz waves

- `SMOOTHING_LOW_WIND = 10`
  - Reason: Middle ground between no-wind and full-wind
  - Result: [To be filled in]

### Ramp Detection
- `MIN_RAMP_PEAKS = 5`
  - Reason: Need enough points to establish trend
  - Result: [To be filled in]

- `MAX_RAMP_PEAKS = 15`
  - Reason: Ramp-up shouldn't be too long
  - Result: [To be filled in]

- `MIN_GROWTH_FACTOR = 1.015`
  - Reason: Amplitude should grow at least 1.5% to be considered ramp
  - Result: Too lenient? Consider 1.02 or 1.05?

### Manual Detection Points
- `FREQ_1_3_HZ_PROBE1_START = 4500`
  - Reason: Eyeballed from full-panel, full-wind, amp02, freq13 test
  - Result: Works for this specific case
  - TODO: Replace with automatic detection

- `FREQ_1_3_HZ_PROBE2_OFFSET = 100`
  - Reason: Wave travel time between probes (100 samples = 0.4 sec)
  - Result: [To be filled in]

---

## Example Future Entry

## 2025-01-31 - Adjusted ramp detection for low-frequency waves

**Changed:** `RAMP.MIN_GROWTH_FACTOR`
- From: 1.015
- To: 1.05
- Reason: 0.65 Hz waves have more gradual ramp-up, need stricter threshold
- Result: Reduced false positives by 30%, but missed 2 valid cases
- Decision: Keep 1.05 for now, add frequency-dependent scaling later

**Changed:** `SIGNAL.BASELINE_SIGMA_FACTOR`
- From: 1.0
- To: 1.5
- Reason: Getting false starts on noisy baseline
- Result: Perfect! No more false starts, still catches all real waves
- Decision: KEEPER - this is the right value

---

## Templates for Common Changes

### Tuning ramp detection
```
Changed: RAMP.MIN_RAMP_PEAKS
From: 5
To: 7
Reason: Too many false positives with 5 peaks
Result: 
- Tested on: [list test cases]
- Success rate: X/Y
- False positives: reduced from A to B
- False negatives: increased from C to D
Decision: [keep/revert/try_something_else]
```

### Adjusting frequency-dependent parameters
```
Changed: Added frequency-specific smoothing
From: Single smoothing window for all frequencies
To: Smoothing = 15 for f > 1.0 Hz, 10 for f < 0.8 Hz
Reason: High-frequency waves need more smoothing
Result: 
- High freq (1.3 Hz): improved SNR by 20%
- Low freq (0.65 Hz): no change (as expected)
Decision: Implement in get_smoothing_window()
```

---

## Testing Protocol

When changing parameters, test on these cases:
- [ ] Full panel, full wind, amp02, freq13
- [ ] Full panel, full wind, amp01, freq065, per15
- [ ] No panel, low wind, amp03, freq065
- [ ] [Add your critical test cases here]

Check:
- [ ] No crashes
- [ ] Wave ranges look sensible (visual check plots)
- [ ] Amplitudes are reasonable (compare with manual measurements)
- [ ] Processing time is acceptable
- [ ] Results are reproducible

---

## Ideas to Try Later

- [ ] Frequency-dependent sigma factor (higher freq = higher threshold?)
- [ ] Adaptive ramp detection based on signal-to-noise ratio
- [ ] Replace manual detection points with ML model
- [ ] Wind-speed-dependent peak detection parameters
- [ ] Automatic outlier detection based on ak (steepness)

---

## Parameter Sensitivity Analysis

Track which parameters matter most:

| Parameter | Impact | Sensitivity | Notes |
|-----------|--------|-------------|-------|
| `BASELINE_SIGMA_FACTOR` | HIGH | MEDIUM | Small changes matter |
| `MIN_RAMP_PEAKS` | MEDIUM | LOW | 5 vs 7 not much difference |
| `SMOOTHING_FULL_WIND` | HIGH | HIGH | Very sensitive |
| `MIN_GROWTH_FACTOR` | LOW | MEDIUM | Only matters for edge cases |

---

## Known Issues & Workarounds

1. **Manual detection for 1.3 Hz and 0.65 Hz**
   - Issue: Hardcoded start positions
   - Workaround: Using eyeballed values
   - TODO: Implement robust automatic detection
   - Target date: [When you get to it]

2. **Short signals (15 periods)**
   - Issue: Not enough data for ramp detection
   - Workaround: [Describe current approach]
   - TODO: [What you plan to do]

---

## Lessons Learned

- **2025-01-30:** Don't trust automatic detection for edge cases - always visual check!
- **2025-01-30:** Wind condition affects optimal smoothing more than I expected
- [Add your insights here]
