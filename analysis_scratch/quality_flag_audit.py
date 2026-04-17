"""
Per-run quality-flag audit — reviewable debug document
======================================================

User-requested (2026-04-17 walk-through of probe_height_wind_findings.md):
produce a debug note that enumerates each flagging layer in
processor.py::_write_quality_flags and walks through how it classified
each run.

For each of the three flagging layers, we list:
  - total runs it evaluated
  - runs it flagged (with triggering metric value)
  - runs that came close to flagging ("near-threshold") — diagnostic for
    catching silent misclassifications
  - pass rate

Layers (in order of application inside _write_quality_flags):
  1. Probe malfunction     → probe_malfunction_{secondary,critical}
  2. Dropout               → dropout_critical
  3. IN-probe low SNR      → in_probe_low_snr

Output:
  analysis_scratch/quality_flag_audit.md
  analysis_scratch/quality_flag_audit_flagged.csv  (machine-readable)

Run:
  /opt/anaconda3/envs/draumkvedet/bin/python analysis_scratch/quality_flag_audit.py
"""

import sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from wavescripts.improved_data_loader import load_analysis_data

# Thresholds — must match processor.py::_write_quality_flags
DROPOUT_THRESHOLD          = 0.02   # >2% of analysis window is NaN → dropout_critical
WAVE_STABILITY_THRESHOLD   = 0.35   # IN wave_stability < 0.35 on nowind → in_probe_low_snr

# "Near-threshold" margins for the diagnostic section
DROPOUT_NEAR_MARGIN        = 0.005  # 0.5–2% reaches diagnostic attention
WAVE_STABILITY_NEAR_MARGIN = 0.10   # 0.35–0.45 reaches diagnostic attention

BASE    = Path(__file__).parent.parent
OUT_MD  = Path(__file__).parent / "quality_flag_audit.md"
OUT_CSV = Path(__file__).parent / "quality_flag_audit_flagged.csv"


def short_path(p: str) -> str:
    """Shorten a wavedata path to folder/basename for readable tables."""
    parts = Path(p).parts
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return Path(p).name


def fmt_float(x, digits=3):
    if pd.isna(x):
        return "—"
    return f"{float(x):.{digits}f}"


# ── 1. Load meta ──────────────────────────────────────────────────────────────
print("Loading metadata…")
dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
meta, _, _, _ = load_analysis_data(*dirs, load_processed=False)
print(f"  {len(meta)} total runs across {len(dirs)} datasets")


# ── 2. Set-up — classify runs for audit ───────────────────────────────────────
# Flagging only applies to runs with a computed analysis window. Identify those.
# Note: different runs may use different probe configs, so 'in_position' /
# 'out_position' can vary. We use these per-run rather than a single pair.
wave = meta[meta["WaveFrequencyInput [Hz]"].notna() &
            (meta["WaveFrequencyInput [Hz]"] > 0)].copy()
nowave = meta[~meta.index.isin(wave.index)].copy()

print(f"  wave runs: {len(wave)}, nowave runs: {len(nowave)}")

# Per-run IN/OUT positions (string columns); needed to probe-column-lookup.
# The probe position strings may be NaN for configs we couldn't resolve.
def _pos(row, key):
    v = row.get(key)
    return None if pd.isna(v) else str(v)


# ── 3. Audit layer 1: probe malfunction ───────────────────────────────────────
# Flag values: "probe_malfunction_critical" if IN/OUT affected,
#              "probe_malfunction_secondary" otherwise.
# Underlying data: per-probe boolean "probe_{pos}_malfunction" columns.
layer1 = {
    "name": "Probe malfunction (stuck segments + DC steps)",
    "flag_values": ["probe_malfunction_critical", "probe_malfunction_secondary"],
    "rationale": (
        "Layer 0b/0c in processor.py. Detects stuck sample runs and DC steps. "
        "Runs where a malfunction segment overlaps the analysis window are "
        "flagged. If the IN or OUT probe is affected → `_critical`, else "
        "`_secondary`."
    ),
}

# Gather probe_X_malfunction columns
pmf_cols = [c for c in meta.columns if c.startswith("probe_") and c.endswith("_malfunction")]
layer1_flagged_runs = meta[
    meta["quality_flag"].isin(layer1["flag_values"])
].copy()

print(f"\nLayer 1 (probe malfunction): {len(layer1_flagged_runs)} flagged")
layer1_detail_rows = []
for _, r in layer1_flagged_runs.iterrows():
    affected = [c.replace("probe_", "").replace("_malfunction", "")
                for c in pmf_cols if bool(r.get(c, False))]
    layer1_detail_rows.append({
        "folder_file":    short_path(r["path"]),
        "flag":           r["quality_flag"],
        "affected_probes": ", ".join(affected) if affected else "(none recorded)",
        "wave_cond":      f"{r.get('WaveFrequencyInput [Hz]', 'nowave')} Hz / "
                          f"{r.get('WaveAmplitudeInput [Volt]', '—')} V",
        "wind":           r.get("WindCondition", "?"),
    })

# ── 4. Audit layer 2: dropout_critical ────────────────────────────────────────
layer2 = {
    "name": "Dropout in analysis window",
    "flag_value": "dropout_critical",
    "threshold": DROPOUT_THRESHOLD,
    "rationale": (
        "cut_samples_{pos} / window_size > 2% at the IN or OUT probe. "
        "Ignored for rows already flagged by Layer 1 (no downgrade)."
    ),
}

# Per-row compute cut_frac at IN and OUT using the appropriate positions
def _dropout_frac(row):
    in_pos  = _pos(row, "in_position")
    out_pos = _pos(row, "out_position")
    start = row.get(f"Computed Probe {in_pos} start") if in_pos else None
    end   = row.get(f"Computed Probe {in_pos} end")   if in_pos else None
    if pd.isna(start) or pd.isna(end):
        return np.nan, np.nan
    win = max(1, int(end) - int(start))
    def _frac(pos):
        if pos is None:
            return np.nan
        n = row.get(f"cut_samples_{pos}")
        if pd.isna(n):
            return np.nan
        return min(int(n), win) / win
    return _frac(in_pos), _frac(out_pos)


dropout_records = []
for _, r in meta.iterrows():
    in_frac, out_frac = _dropout_frac(r)
    dropout_records.append({
        "path":     r["path"],
        "flag":     r["quality_flag"],
        "in_frac":  in_frac,
        "out_frac": out_frac,
        "max_frac": np.nanmax([in_frac, out_frac]) if not (pd.isna(in_frac) and pd.isna(out_frac)) else np.nan,
    })
drop_df = pd.DataFrame(dropout_records)
drop_df["near_threshold"] = (
    drop_df["max_frac"].notna() &
    (drop_df["max_frac"] >= DROPOUT_THRESHOLD - DROPOUT_NEAR_MARGIN) &
    (drop_df["max_frac"] < DROPOUT_THRESHOLD) &
    (drop_df["flag"] == "ok")  # not already flagged
)

layer2_flagged_runs = meta[meta["quality_flag"] == "dropout_critical"].copy()
layer2_near_runs    = drop_df[drop_df["near_threshold"]].copy()

print(f"Layer 2 (dropout_critical): {len(layer2_flagged_runs)} flagged; "
      f"{len(layer2_near_runs)} near-threshold (0.5-2% dropout)")

# ── 5. Audit layer 3: in_probe_low_snr ────────────────────────────────────────
layer3 = {
    "name": "IN-probe low SNR (nowind wave runs only)",
    "flag_value": "in_probe_low_snr",
    "threshold": WAVE_STABILITY_THRESHOLD,
    "rationale": (
        "Nowind wave runs with Probe {in_pos} wave_stability < 0.35 → "
        "PCHIP-reconstructed signal flattened; FFT amplitude unreliable. "
        "Threshold empirical (see processor.py:1607)."
    ),
}

ws_records = []
for _, r in meta.iterrows():
    in_pos = _pos(r, "in_position")
    ws = r.get(f"Probe {in_pos} wave_stability") if in_pos else np.nan
    ws_records.append({
        "path":         r["path"],
        "flag":         r["quality_flag"],
        "wind":         r.get("WindCondition"),
        "has_wave":     pd.notna(r.get("WaveFrequencyInput [Hz]"))
                         and r.get("WaveFrequencyInput [Hz]", 0) > 0,
        "in_position":  in_pos,
        "wave_stability": ws,
    })
ws_df = pd.DataFrame(ws_records)
ws_df["near_threshold"] = (
    ws_df["wave_stability"].notna() &
    (ws_df["wave_stability"] >= WAVE_STABILITY_THRESHOLD) &
    (ws_df["wave_stability"] < WAVE_STABILITY_THRESHOLD + WAVE_STABILITY_NEAR_MARGIN) &
    (ws_df["wind"] == "no") &
    ws_df["has_wave"] &
    (ws_df["flag"] == "ok")
)

layer3_flagged_runs = meta[meta["quality_flag"] == "in_probe_low_snr"].copy()
layer3_near_runs    = ws_df[ws_df["near_threshold"]].copy()

print(f"Layer 3 (in_probe_low_snr): {len(layer3_flagged_runs)} flagged; "
      f"{len(layer3_near_runs)} near-threshold (ws 0.35-0.45, nowind wave)")


# ── 6. Full-meta quality-flag distribution ────────────────────────────────────
flag_counts = meta["quality_flag"].value_counts(dropna=False)
print("\nquality_flag distribution:")
for k, v in flag_counts.items():
    print(f"  {k}: {v}")


# ── 7. Flagged CSV (machine-readable) ─────────────────────────────────────────
all_flagged = meta[meta["quality_flag"] != "ok"].copy()
csv_cols = [
    "path", "quality_flag", "WindCondition", "WaveFrequencyInput [Hz]",
    "WaveAmplitudeInput [Volt]", "in_position", "out_position",
    "IN wave_stability", "OUT wave_stability",
]
for c in csv_cols[:]:
    if c not in all_flagged.columns:
        csv_cols.remove(c)
all_flagged[csv_cols].to_csv(OUT_CSV, index=False)
print(f"\nFlagged runs CSV → {OUT_CSV.relative_to(BASE)}")


# ── 8. Write markdown ─────────────────────────────────────────────────────────
def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(none)_\n"
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(fmt_float(v))
            elif pd.isna(v):
                cells.append("—")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows) + "\n"


lines = []
lines.append("# Quality-flag audit (per-run)")
lines.append("")
lines.append(f"Generated: {pd.Timestamp.utcnow().isoformat()[:19]}Z")
lines.append("")
lines.append("Source: `analysis_scratch/quality_flag_audit.py` → this doc")
lines.append("")
lines.append(f"- Total runs: **{len(meta)}**")
lines.append(f"- Wave runs: **{len(wave)}**  ·  nowave runs: **{len(nowave)}**")
lines.append("")
lines.append("## quality_flag distribution")
lines.append("")
lines.append("| flag | count |")
lines.append("|---|---|")
for k, v in flag_counts.items():
    lines.append(f"| `{k}` | {v} |")
lines.append("")

# ── Layer 1 ───────────────────────────────────────────────────────────────────
lines.append("## Layer 1 — Probe malfunction")
lines.append("")
lines.append(layer1["rationale"])
lines.append("")
lines.append(f"Flag values: `{', '.join(layer1['flag_values'])}`")
lines.append(f"Runs flagged: **{len(layer1_flagged_runs)}**")
lines.append("")
if layer1_detail_rows:
    lines.append("### Flagged runs")
    lines.append("")
    lines.append(_md_table(pd.DataFrame(layer1_detail_rows)))
else:
    lines.append("_No runs flagged by this layer._")
    lines.append("")

# ── Layer 2 ───────────────────────────────────────────────────────────────────
lines.append("## Layer 2 — Dropout in analysis window")
lines.append("")
lines.append(layer2["rationale"])
lines.append("")
lines.append(f"Threshold: `cut_samples / window_size > {DROPOUT_THRESHOLD:.2f}` "
             f"at IN or OUT probe")
lines.append(f"Runs flagged as `dropout_critical`: **{len(layer2_flagged_runs)}**")
lines.append(f"Runs near-threshold (max_frac in [{DROPOUT_THRESHOLD-DROPOUT_NEAR_MARGIN:.3f}, "
             f"{DROPOUT_THRESHOLD:.2f})): **{len(layer2_near_runs)}**")
lines.append("")
if not layer2_flagged_runs.empty:
    lines.append("### Flagged runs")
    lines.append("")
    flagged_display = layer2_flagged_runs.copy()
    # Add max_frac by looking up from drop_df
    frac_map = drop_df.set_index("path")["max_frac"].to_dict()
    flagged_display["max_cut_frac"] = flagged_display["path"].map(frac_map)
    flagged_display["folder_file"] = flagged_display["path"].map(short_path)
    lines.append(_md_table(flagged_display[[
        "folder_file", "max_cut_frac",
        "WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]", "WindCondition",
    ]].rename(columns={"max_cut_frac": "max_cut_frac",
                       "WaveFrequencyInput [Hz]": "freq",
                       "WaveAmplitudeInput [Volt]": "amp",
                       "WindCondition": "wind"})))
else:
    lines.append("_No runs flagged by this layer._")
    lines.append("")

if not layer2_near_runs.empty:
    lines.append("### Near-threshold (diagnostic — not flagged)")
    lines.append("")
    lines.append("Runs with moderate NaN-dropout that passed the 2% threshold. "
                 "Worth a look if a trend in the data points at these.")
    lines.append("")
    near_display = layer2_near_runs.copy()
    near_display["folder_file"] = near_display["path"].map(short_path)
    lines.append(_md_table(near_display[[
        "folder_file", "in_frac", "out_frac", "max_frac",
    ]].rename(columns={"in_frac": "IN_cut_frac",
                       "out_frac": "OUT_cut_frac",
                       "max_frac": "max_frac"})))

# ── Layer 3 ───────────────────────────────────────────────────────────────────
lines.append("## Layer 3 — IN-probe low SNR (nowind wave runs)")
lines.append("")
lines.append(layer3["rationale"])
lines.append("")
lines.append(f"Threshold: `Probe {{in_pos}} wave_stability < {WAVE_STABILITY_THRESHOLD:.2f}` "
             f"(nowind wave runs only)")
lines.append(f"Runs flagged as `in_probe_low_snr`: **{len(layer3_flagged_runs)}**")
lines.append(f"Runs near-threshold (ws in [{WAVE_STABILITY_THRESHOLD:.2f}, "
             f"{WAVE_STABILITY_THRESHOLD + WAVE_STABILITY_NEAR_MARGIN:.2f})): "
             f"**{len(layer3_near_runs)}**")
lines.append("")
if not layer3_flagged_runs.empty:
    lines.append("### Flagged runs")
    lines.append("")
    flagged_display = layer3_flagged_runs.copy()
    ws_map = ws_df.set_index("path")["wave_stability"].to_dict()
    flagged_display["wave_stability"] = flagged_display["path"].map(ws_map)
    flagged_display["folder_file"] = flagged_display["path"].map(short_path)
    lines.append(_md_table(flagged_display[[
        "folder_file", "wave_stability",
        "WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]",
    ]].rename(columns={
        "WaveFrequencyInput [Hz]": "freq",
        "WaveAmplitudeInput [Volt]": "amp",
    })))
else:
    lines.append("_No runs flagged by this layer._")
    lines.append("")

if not layer3_near_runs.empty:
    lines.append("### Near-threshold (diagnostic — not flagged)")
    lines.append("")
    lines.append("Nowind wave runs with IN wave_stability in [0.35, 0.45). "
                 "These pass the quality check but are the next candidates for "
                 "scrutiny if a systematic issue is suspected.")
    lines.append("")
    near_display = layer3_near_runs.copy()
    near_display["folder_file"] = near_display["path"].map(short_path)
    lines.append(_md_table(near_display[[
        "folder_file", "wave_stability", "in_position",
    ]]))

# ── Summary ───────────────────────────────────────────────────────────────────
lines.append("## Summary")
lines.append("")
total_flagged = int((meta["quality_flag"] != "ok").sum())
lines.append(f"- {total_flagged}/{len(meta)} runs have a non-'ok' quality_flag")
lines.append(f"- Breakdown above; each flag was applied by exactly one layer "
             f"(layers are evaluated in order; later layers do not downgrade)")
lines.append(f"- Flagged runs CSV: `analysis_scratch/quality_flag_audit_flagged.csv`")
lines.append("")
lines.append("### What the default filter excludes")
lines.append("")
lines.append("By default, `apply_experimental_filters(combined_meta, ...)` keeps:")
lines.append("- `ok`")
lines.append("- `probe_malfunction_secondary` (IN/OUT both fine, auxiliary probe broken)")
lines.append("")
lines.append("And excludes:")
lines.append("- `probe_malfunction_critical`")
lines.append("- `dropout_critical`")
lines.append("- `in_probe_low_snr`")
lines.append("")
lines.append("Override: pass `filters['quality_flag'] = 'all'` in plotvariables to "
             "include everything.")
lines.append("")
lines.append("### Re-generate")
lines.append("")
lines.append("```bash")
lines.append("python analysis_scratch/quality_flag_audit.py")
lines.append("```")

OUT_MD.write_text("\n".join(lines) + "\n")
print(f"Markdown → {OUT_MD.relative_to(BASE)}")
print(f"\nDone. {total_flagged}/{len(meta)} runs flagged.")
