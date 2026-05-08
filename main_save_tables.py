"""Render thesis tables from precomputed CSV + meta.json sidecars.

RENDER-ONLY by design — cells in this file NEVER invoke the data
scripts. Editing column_headers / cell_format / row_groups in a cell
and re-running it produces a fresh .tex in ~50 ms against whatever
CSV + meta sidecars are currently on disk. Pipeline reloads do not
happen here.

To regenerate a CSV + meta after editing the data script (e.g.
changing a probe label, bumping a column), invoke the script
yourself:

    python analysis_scratch/<name>_table.py

Then re-run the cell. The data script is invoked manually so the
cost of the data layer is paid only when explicitly asked. If a CSV
or meta sidecar is missing on disk when a cell runs, the cell prints
the regen command and skips rendering (no error).

Sister script to main_save_figures.py (figures) and main_save_extras.py
(not-in-thesis figures). Captions for ALL thesis tables live here in
TABLE_CAPTIONS / TABLE_CAPTIONS_SHORT (single source of truth) and are
written to output/.table_captions.json on import — including tables
whose data still renders from main_save_figures.py cells (those scripts
read this JSON via _lookup_central_caption(json_path=...)).

Currently wired (rendered here, two-step pattern):
  ch04_probe_noise_floor_table
  ch04_parallel_probe_psd_agreement_simple
  ch04_window_intervals
  ch04_wind_pre_paddle_table
  ch04_plateau_values
  ch05_damping_freq_table
  ch05_mooring_focus_at_1_3hz_table

Captions-only (rendered from main_save_figures.py cells):
  ch04_window_choice_nowind
  ch04_window_choice_fullwind
  ch04_tidsvindu
  ch04_wind_setup_baseline_table
  ch05_wind_effect_table
  ch05_wind_effect_table_by_amp
  ch05_transmission_wind_ratios
  ch05_transmission_wind_amplitudes
"""

# %%
import json
import os
import sys
import warnings
from pathlib import Path

BASE = (Path(__file__).resolve().parent
        if "__file__" in globals() else Path.cwd())
os.chdir(BASE); sys.path.insert(0, str(BASE))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from wavescripts.table_render import render_table

# ══════════════════════════════════════════════════════════════════════════════
# TABLE CAPTIONS — single source of truth for ALL thesis tables.
# Mirrors FIGURE_CAPTIONS in main_save_figures.py. Empty strings → renderer
# emits "% TODO: write caption" inside the body's \caption{}; missing key
# in the short dict → no [short] arg.
#
# This dict covers BOTH (a) tables fully wired here (two-step CSV+meta render
# below) and (b) tables whose data still lives in main_save_figures.py cells
# but whose captions have been centralised here. The (b) scripts call
# _lookup_central_caption(name, json_path=Path("output/.table_captions.json"))
# so they read these strings via the JSON cache written below.
# ══════════════════════════════════════════════════════════════════════════════
TABLE_CAPTIONS = {
    # ── CHAPTER 04 — METHODOLOGY ─────────────────────────────────────────────
    "ch04_probe_noise_floor_table":              "Oversikt over støygulvet til prober ved innledende og endelig oppsett.",   # TODO: write caption
    "ch04_parallel_probe_psd_agreement_simple":  "Forskjellen mellom parallelle prober",   # TODO: caption — "the two parallel probes agree to within ~2% at every thesis frequency"
    "ch04_window_intervals":                     "",
    "ch04_window_choice_nowind":                 "",
    "ch04_window_choice_fullwind":               "",
    "ch04_plateau_values":                       "Beregnet amplitude fra hvert tidsvindu. Samlet for alle tre amplituder.Inngående og utgående. Transmisjonskoeffisient, og dens standardavvik.",
    "ch04_tidsvindu":                            "Frekvensenes tidsvinduer",
    "ch04_wind_pre_paddle_table":                "Vindspekteret fra lange målinger sammenliknet med 3-sekundersmålinger fra hver kjøring",
    "ch04_wind_setup_baseline_table":            "Målt endring i vannstand ved å se på utgående probe. Fire datasett.",

    # ── CHAPTER 05 — RESULTS ─────────────────────────────────────────────────
    "ch05_damping_freq_table":           "",   # TODO: caption — per-amp K_t,uten, K_t,vind, ΔK_t across 1.3–1.6 Hz, mirrors ch05_damping_freq layout
    "ch05_wind_effect_table":            "Transmisjon for våre utvalgte bølger",
    "ch05_wind_effect_table_by_amp":     "Transmisjon for våre utvalgte bølger",   # TODO: caption — same data as ch05_wind_effect_table, sorted amp-outer / freq-inner
    "ch05_transmission_wind_ratios":     "",
    "ch05_transmission_wind_amplitudes": "",
    "ch05_mooring_focus_at_1_3hz_table": r"Tall til figur \ref{fig:ch05_mooring_focus_at_1_3hz_ka}. Transmisjon for panelrekken fortøyd på ulike måter. Merk: kun for \qty{1.3}{\hertz}. Antall (n) kjøringer.",
}
TABLE_CAPTIONS_SHORT = {
    # ── CHAPTER 04 ───────────────────────────────────────────────────────────
    "ch04_probe_noise_floor_table":              "",   # TODO: short caption
    "ch04_parallel_probe_psd_agreement_simple":  "",
    "ch04_window_intervals":                     "",
    "ch04_window_choice_nowind":                 "",
    "ch04_window_choice_fullwind":               "",
    "ch04_plateau_values":                       "",
    "ch04_tidsvindu":                            "Frekvensenes tidsvindu",
    "ch04_wind_pre_paddle_table":                "",
    "ch04_wind_setup_baseline_table":            "",

    # ── CHAPTER 05 ───────────────────────────────────────────────────────────
    "ch05_damping_freq_table":           "",
    "ch05_wind_effect_table":            "",
    "ch05_wind_effect_table_by_amp":     "Transmisjon for våre utvalgte bølger",
    "ch05_transmission_wind_ratios":     "",
    "ch05_transmission_wind_amplitudes": "",
    "ch05_mooring_focus_at_1_3hz_table": "Moring vs panelretning. 1,3 Hz",
}

# Persist for any downstream reader (e.g. _lookup_central_caption with the
# json_path kwarg). Gitignored — source of truth is the dicts above.
_TABLE_CAPTIONS_JSON = BASE / "output" / ".table_captions.json"
_TABLE_CAPTIONS_JSON.parent.mkdir(parents=True, exist_ok=True)
_TABLE_CAPTIONS_JSON.write_text(
    json.dumps(
        {"full": TABLE_CAPTIONS, "short": TABLE_CAPTIONS_SHORT},
        indent=2, ensure_ascii=False,
    ),
    encoding="utf-8",
)


def _render_with_caption_short(
    meta_path: Path,
    short_caption: str,
    render_fn,
) -> None:
    """Render the .tex from existing CSV + meta sidecars. Never fires the data script.

    Cells in this file are RENDER-ONLY by design. Editing column_headers
    / cell_format / row_groups in the cell and re-running it never
    triggers a pipeline reload — render runs in ~50 ms against the CSV
    on disk.

    To regenerate the CSV + meta after editing the data SCRIPT (e.g.
    changing a probe label, bumping a column), invoke the script
    yourself:

        python <script_rel from meta.json>

    Then re-run this cell.

    Behaviour:
      1. Derives `csv_path` from `meta_path` by sibling-file convention
         (replaces `.meta.json` with `.csv`).
      2. If either sidecar is missing, prints a warning with the
         script-regen command (read from meta.json's `script` field
         when available) and returns without rendering.
      3. Otherwise patches `caption_short` into meta.json on disk just
         long enough for `render_fn()` to read it, then reverts the
         file via try/finally so meta.json stays unchanged.
    """
    csv_path = meta_path.parent / meta_path.name.replace(".meta.json", ".csv")
    missing = [p.name for p in (csv_path, meta_path) if not p.exists()]
    if missing:
        script_hint = ""
        if meta_path.exists():
            try:
                _m = json.loads(meta_path.read_text(encoding="utf-8"))
                if _m.get("script"):
                    script_hint = f"     Run: python {_m['script']}"
            except Exception:
                pass
        print(f"  ⚠ {csv_path.stem}: missing sidecar(s) {missing}")
        if script_hint:
            print(script_hint)
        return

    if not short_caption:
        render_fn()
        return
    raw = meta_path.read_text(encoding="utf-8")
    meta = json.loads(raw)
    if meta.get("caption_short") == short_caption:
        render_fn()
        return
    meta["caption_short"] = short_caption
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    try:
        render_fn()
    finally:
        meta_path.write_text(raw, encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# Shared cell formatter — used by ≥ 2 tables. Per-table-only formatters live
# inline in their respective cell.
# ══════════════════════════════════════════════════════════════════════════════
def _fmt_signed(x: float, decimals: int = 3) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:+.{decimals}f}"


# %%
# TABLE_INDEX
# ═══════════════════════════════════════════════════════════════════════════════
#   ch04_probe_noise_floor_table             [RENDER] ✓  3σ noise per probe — innledende vs endelig
#   ch04_parallel_probe_psd_agreement_simple [RENDER] ✓  Δ% (far−wall) per thesis freq
#   ch04_window_intervals                    [RENDER] ✓  H&G theoretical window intervals per freq
#   ch04_wind_pre_paddle_table               [RENDER] ✓  σ_η long vs 3 s pre-paddle per probe
#   ch05_damping_freq_table          [RENDER] ✓  Per-amp K_t,uten/K_t,vind/ΔK_t at 1.3–1.6 Hz
#   ch05_mooring_focus_at_1_3hz_table [RENDER] ✓  Mooring × panel × wind transmission at 1.30 Hz
#   ch04_plateau_values              [RENDER] ✓  A_in/A_out/K_t per (amp, freq, wind), all 3 amplitudes
# ═══════════════════════════════════════════════════════════════════════════════


# %%
# [DATA: RENDER]  — analysis_scratch/probe_noise_floor_table.py
"""
── CH04 § 1 — Probe noise-floor table (innledende vs endelig) ──────────────
Per probe position, 3σ stillwater detection threshold for the initial
hardware config (h272 / high range) vs the final canon config (h100 / low
range), plus the improvement ratio. Single block of 4 rows (one per probe).
"""
_NAME = "ch04_probe_noise_floor_table"
_CSV  = Path(f"output/TABLES/data/{_NAME}.csv")
_META = Path(f"output/TABLES/data/{_NAME}.meta.json")
_TEX  = Path(f"output/TABLES/{_NAME}.tex")

def _fmt_mm_2dp(col: str):
    def _impl(row: pd.Series) -> str:
        v = row[col]
        if pd.isna(v):
            return "—"
        return rf"$\num{{{v:.2f}}}$"
    return _impl


def _fmt_ratio_x(col: str):
    def _impl(row: pd.Series) -> str:
        v = row[col]
        if pd.isna(v):
            return "—"
        return rf"$\num{{{v:.1f}}}\times$"
    return _impl


_cell_format = {
    "probe_label":            lambda r: str(r["probe_label"]),
    "thr3sigma_initial_mm":   _fmt_mm_2dp("thr3sigma_initial_mm"),
    "thr3sigma_final_mm":     _fmt_mm_2dp("thr3sigma_final_mm"),
    "ratio_init_over_final":  _fmt_ratio_x("ratio_init_over_final"),
}

_columns = [
    "probe_label",
    "thr3sigma_initial_mm",
    "thr3sigma_final_mm",
    "ratio_init_over_final",
]
# Two-row header — second row stuffed onto the last column header so the
# joined `& `-output and trailing `\\` produce a tidy 2-line header.
_column_headers = [
    "Probe",
    "Innledende",
    "Endelig",
    (
        "Forbedring \\\\\n"
        "         &\n"
        "      $3\\sigma$ [\\unit{\\milli\\metre}] &\n"
        "      $3\\sigma$ [\\unit{\\milli\\metre}] &\n"
        "         "
    ),
]

_render_with_caption_short(
    _META,
    TABLE_CAPTIONS_SHORT.get(_NAME) or "",
    lambda: render_table(
        csv_path       = _CSV,
        meta_path      = _META,
        out_tex_path   = _TEX,
        columns        = _columns,
        column_headers = _column_headers,
        column_spec    = "lccc",
        cell_format    = _cell_format,
        row_groups     = [(None, lambda df: df)],
        label          = f"tab:{_NAME}",
        caption        = TABLE_CAPTIONS.get(_NAME) or None,
        short_caption  = TABLE_CAPTIONS_SHORT.get(_NAME) or None,
    ),
)
print(f"   TEX → {_TEX}")


# %%
# [DATA: RENDER]  — analysis_scratch/parallel_probe_psd_agreement_simple.py
"""
── CH04 § 3 — Parallel-probe PSD agreement (simple) ─────────────────────────
Per thesis frequency, N runs, mean amplitude, signed mean Δ% between
9373/170 (wall) and 9373/340 (far). One block, four rows.
"""
_NAME = "ch04_parallel_probe_psd_agreement_simple"
_CSV  = Path(f"output/TABLES/data/{_NAME}.csv")
_META = Path(f"output/TABLES/data/{_NAME}.meta.json")
_TEX  = Path(f"output/TABLES/{_NAME}.tex")

def _fmt_freq_2dp(row: pd.Series) -> str:
    return rf"\num{{{row['freq']:.2f}}}"


def _fmt_int_n(row: pd.Series) -> str:
    return rf"\num{{{int(row['n'])}}}"


def _fmt_amp_2dp(row: pd.Series) -> str:
    v = row["mean_amp"]
    if pd.isna(v):
        return "n/a"
    return rf"\num{{{v:.2f}}}"


def _fmt_signed_pct_2dp(row: pd.Series) -> str:
    v = row["diff_pct"]
    if pd.isna(v):
        return "n/a"
    return rf"\num{{{v:+.2f}}}"


_cell_format = {
    "freq":     _fmt_freq_2dp,
    "n":        _fmt_int_n,
    "mean_amp": _fmt_amp_2dp,
    "diff_pct": _fmt_signed_pct_2dp,
}

_columns        = ["freq", "n", "mean_amp", "diff_pct"]
_column_headers = [
    r"$f$ [\unit{\hertz}]",
    r"$N$",
    r"$\langle A \rangle$ [\unit{\milli\metre}]",
    r"$\Delta$ (fjern$-$nær) [\%]",
]

_render_with_caption_short(
    _META,
    TABLE_CAPTIONS_SHORT.get(_NAME) or "",
    lambda: render_table(
        csv_path       = _CSV,
        meta_path      = _META,
        out_tex_path   = _TEX,
        columns        = _columns,
        column_headers = _column_headers,
        column_spec    = "cccc",
        cell_format    = _cell_format,
        row_groups     = [(None, lambda df: df)],
        label          = f"tab:{_NAME}",
        caption        = TABLE_CAPTIONS.get(_NAME) or None,
        short_caption  = TABLE_CAPTIONS_SHORT.get(_NAME) or None,
    ),
)
print(f"   TEX → {_TEX}")


# %%
# [DATA: RENDER]  — analysis_scratch/window_intervals_table.py
"""
── CH04 § 4 — ( ! note exact "H&G"-numbers is not in use anymore.) theoretical window intervals per thesis frequency ────────
Wide layout: each thesis frequency is a COLUMN, the rows are
(Innkommende [s], Utgående [s], samples per period). The CSV is per-freq;
this cell pivots it into the wide form via per-row cell formatters that
read the relevant freq-column out of the long CSV.
"""
_NAME = "ch04_window_intervals"
_CSV  = Path(f"output/TABLES/data/{_NAME}.csv")
_META = Path(f"output/TABLES/data/{_NAME}.meta.json")
_TEX  = Path(f"output/TABLES/{_NAME}.tex")

_WIN_FREQS = [1.3, 1.4, 1.5, 1.6]
_WIN_FREQ_COLS = [f"f_{f:.1f}" for f in _WIN_FREQS]


def _fmt_in_range(freq_col: str):
    def _impl(row: pd.Series) -> str:
        s = row[f"{freq_col}_in_start_s"]
        e = row[f"{freq_col}_in_end_s"]
        return rf"\tabnumrange{{{s:.1f}}}{{{e:.1f}}}"
    return _impl


def _fmt_out_range(freq_col: str):
    def _impl(row: pd.Series) -> str:
        s = row[f"{freq_col}_out_start_s"]
        e = row[f"{freq_col}_out_end_s"]
        return rf"\tabnumrange{{{s:.1f}}}{{{e:.1f}}}"
    return _impl


def _fmt_spp(freq_col: str):
    def _impl(row: pd.Series) -> str:
        v = row[f"{freq_col}_samples_per_period"]
        return rf"$\num{{{v:.10f}}}$"
    return _impl


# Each row in the CSV is one thesis row (label + per-freq cells). Build
# cell formatters that look up the {row label, freq column} cell.
def _fmt_label(row: pd.Series) -> str:
    return str(row["row_label"])


def _fmt_value_for_kind(freq_col: str):
    def _impl(row: pd.Series) -> str:
        kind = row["kind"]
        v = row[f"{freq_col}_value"]
        if kind == "in_range":
            s, e = v.split("|")
            return rf"\tabnumrange{{{float(s):.1f}}}{{{float(e):.1f}}}"
        if kind == "out_range":
            s, e = v.split("|")
            return rf"\tabnumrange{{{float(s):.1f}}}{{{float(e):.1f}}}"
        if kind == "spp":
            return rf"$\num{{{float(v):.10f}}}$"
        return str(v)
    return _impl


_cell_format = {"row_label": _fmt_label}
for _fc in _WIN_FREQ_COLS:
    _cell_format[f"{_fc}_value"] = _fmt_value_for_kind(_fc)

_columns        = ["row_label", *[f"{_fc}_value" for _fc in _WIN_FREQ_COLS]]
_column_headers = [
    r"Frekvens [\unit{\hertz}]",
    *[rf"$\num{{{f}}}$" for f in _WIN_FREQS],
]

_render_with_caption_short(
    _META,
    TABLE_CAPTIONS_SHORT.get(_NAME) or "",
    lambda: render_table(
        csv_path       = _CSV,
        meta_path      = _META,
        out_tex_path   = _TEX,
        columns        = _columns,
        column_headers = _column_headers,
        column_spec    = "lcccc",
        cell_format    = _cell_format,
        row_groups     = [(None, lambda df: df)],
        label          = f"tab:{_NAME}",
        caption        = TABLE_CAPTIONS.get(_NAME) or None,
        short_caption  = TABLE_CAPTIONS_SHORT.get(_NAME) or None,
    ),
)
print(f"   TEX → {_TEX}")


# %%
# [DATA: RENDER]  — analysis_scratch/wind_pre_paddle_table.py
"""
── CH04 § 4q — Pre-paddle wind summary (long-run vs 3 s) per probe ─────────
Per probe row: long-run σ_η, 3 s mean σ_η, Δ%, 3 s 1σ scatter. One block,
four rows.
"""
_NAME = "ch04_wind_pre_paddle_table"
_CSV  = Path(f"output/TABLES/data/{_NAME}.csv")
_META = Path(f"output/TABLES/data/{_NAME}.meta.json")
_TEX  = Path(f"output/TABLES/{_NAME}.tex")



def _fmt_mm(col: str, decimals: int = 2):
    def _impl(row: pd.Series) -> str:
        v = row[col]
        if pd.isna(v):
            return "—"
        return rf"$\num{{{v:.{decimals}f}}}$"
    return _impl


def _fmt_signed_pct(col: str, decimals: int = 1):
    def _impl(row: pd.Series) -> str:
        v = row[col]
        if pd.isna(v):
            return "—"
        sign = "+" if v >= 0 else "-"
        return rf"${sign}\num{{{abs(v):.{decimals}f}}}$"
    return _impl


_cell_format = {
    "probe":               lambda r: str(r["probe"]),
    "sigma_long_mm":       _fmt_mm("sigma_long_mm", 2),
    "sigma_3s_mm":         _fmt_mm("sigma_3s_mm",   2),
    "delta_pct":           _fmt_signed_pct("delta_pct", 1),
    "sigma_3s_scatter_mm": _fmt_mm("sigma_3s_scatter_mm", 2),
}

_columns        = [
    "probe", "sigma_long_mm", "sigma_3s_mm", "delta_pct",
    "sigma_3s_scatter_mm",
]
_column_headers = [
    "Probe",
    r"$\sigma_\eta$ (lang) [\unit{\milli\metre}]",
    r"$\sigma_\eta$ (\qty{3}{\second}) [\unit{\milli\metre}]",
    r"$\Delta$ [\%]",
    r"$\sigma_\eta$ (\qty{3}{\second}, $1\sigma$) [\unit{\milli\metre}]",
]

_render_with_caption_short(
    _META,
    TABLE_CAPTIONS_SHORT.get(_NAME) or "",
    lambda: render_table(
        csv_path       = _CSV,
        meta_path      = _META,
        out_tex_path   = _TEX,
        columns        = _columns,
        column_headers = _column_headers,
        column_spec    = "lcccc",
        cell_format    = _cell_format,
        row_groups     = [(None, lambda df: df)],
        label          = f"tab:{_NAME}",
        caption        = TABLE_CAPTIONS.get(_NAME) or None,
        short_caption  = TABLE_CAPTIONS_SHORT.get(_NAME) or None,
    ),
)
print(f"   TEX → {_TEX}")


# %%
# [DATA: RENDER]  — analysis_scratch/damping_freq_table.py
"""
── CH05 § 1b — Damping-vs-frequency table (companion to ch05_damping_freq) ─
Per amplitude tier (A1/A2/A3), tabulates K_t at no-wind, K_t at full-wind,
and ΔK_t across the four thesis frequencies. Three row-blocks mirror the
three stacked subfigures of ch05_damping_freq.
"""
_NAME = "ch05_damping_freq_table"
_CSV  = Path(f"output/TABLES/data/{_NAME}.csv")
_META = Path(f"output/TABLES/data/{_NAME}.meta.json")
_TEX  = Path(f"output/TABLES/{_NAME}.tex")


_THESIS_FREQS = [1.3, 1.4, 1.5, 1.6]
_THESIS_AMPS  = [0.10, 0.20, 0.30]
_FREQ_COLS    = [f"f_{f:.1f}" for f in _THESIS_FREQS]


def _fmt_unsigned(x: float, decimals: int = 3) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:.{decimals}f}"


def _fmt_value_cell(freq_col: str):
    def _impl(row: pd.Series) -> str:
        v = row[freq_col]
        if row["kind"] == "delta":
            return _fmt_signed(v, 3)
        return _fmt_unsigned(v, 3)
    return _impl


_cell_format = {
    "amp_label_display": lambda r: str(r["amp_label_display"]),
    "kind_label":        lambda r: str(r["kind_label"]),
}
for _fc in _FREQ_COLS:
    _cell_format[_fc] = _fmt_value_cell(_fc)

_columns        = ["amp_label_display", "kind_label"] + _FREQ_COLS
_column_headers = ["", "", *[f"{f:.1f}\\,Hz" for f in _THESIS_FREQS]]

_row_groups: list[tuple[str | None, callable]] = []
for _amp in _THESIS_AMPS:
    _row_groups.append(
        (None, (lambda a: lambda df: df[np.isclose(df["amp_volt"], a)])(_amp))
    )

_render_with_caption_short(
    _META,
    TABLE_CAPTIONS_SHORT.get(_NAME) or "",
    lambda: render_table(
        csv_path       = _CSV,
        meta_path      = _META,
        out_tex_path   = _TEX,
        columns        = _columns,
        column_headers = _column_headers,
        column_spec    = "ll cccc",
        cell_format    = _cell_format,
        row_groups     = _row_groups,
        label          = f"tab:{_NAME}",
        caption        = TABLE_CAPTIONS.get(_NAME) or None,
        short_caption  = TABLE_CAPTIONS_SHORT.get(_NAME) or None,
    ),
)
print(f"   TEX → {_TEX}")


# %%
# [DATA: RENDER]  — analysis_scratch/mooring_focus_at_1_3hz_table.py
"""
── CH05 § 4b — Mooring + panelretning at 1.30 Hz: companion table ───────────
Hard numbers for ch05_mooring_focus_at_1_3hz_ka. Same data, same scope (1.30
Hz only, panels ∈ {full, reverse}, moorings ∈ {below_90, above_50}).
"""
_NAME = "ch05_mooring_focus_at_1_3hz_table"
_CSV  = Path(f"output/TABLES/data/{_NAME}.csv")
_META = Path(f"output/TABLES/data/{_NAME}.meta.json")
_TEX  = Path(f"output/TABLES/{_NAME}.tex")

_THESIS_AMPS = [0.10, 0.20, 0.30]


def _fmt_ratio(x: float, decimals: int = 3) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:.{decimals}f}"


def _fmt_kt_n_factory(kt_col: str, n_col: str):
    """Compound cell: '0.658\\,(23)' built from Kt + n columns."""
    def _impl(row: pd.Series) -> str:
        k = row[kt_col]
        n = row[n_col]
        if pd.isna(k):
            return "—"
        n_str = "—" if pd.isna(n) else f"{int(n)}"
        return f"{k:.3f}\\,({n_str})"
    return _impl


def _fmt_str_or_blank(col: str):
    def _impl(row: pd.Series) -> str:
        v = row[col]
        if pd.isna(v):
            return ""
        return str(v)
    return _impl


_cell_format = {
    "amp_label_display": _fmt_str_or_blank("amp_label_display"),
    "panel_label":       _fmt_str_or_blank("panel_label"),
    "mooring_label":     _fmt_str_or_blank("mooring_label"),
    "Kt_nw_n":           _fmt_kt_n_factory("Kt_nw", "n_nw"),
    "Kt_fw_n":           _fmt_kt_n_factory("Kt_fw", "n_fw"),
    "Delta_Kt":          lambda r: _fmt_signed(r["Delta_Kt"], 3),
    "ratio_Kt":          lambda r: _fmt_ratio(r["ratio_Kt"], 3),
    "ratio_D":           lambda r: _fmt_ratio(r["ratio_D"],  3),
}

_columns = [
    "amp_label_display", "panel_label", "mooring_label",
    "Kt_nw_n", "Kt_fw_n",
    "Delta_Kt", "ratio_Kt", "ratio_D",
]
_column_headers = [
    "Amp", "Panel", "Mooring",
    r"$K_{t,\text{uten}}\,(n)$",
    r"$K_{t,\text{vind}}\,(n)$",
    r"$\Delta K_t$",
    r"$K_{t,\text{vind}}/K_{t,\text{uten}}$",
    r"$D_{\text{vind}}/D_{\text{uten}}$",
]

_row_groups: list[tuple[str | None, callable]] = []
for _amp in _THESIS_AMPS:
    _row_groups.append(
        (None, (lambda a: lambda df: df[np.isclose(df["amp_v"], a)])(_amp))
    )

_render_with_caption_short(
    _META,
    TABLE_CAPTIONS_SHORT.get(_NAME) or "",
    lambda: render_table(
        csv_path       = _CSV,
        meta_path      = _META,
        out_tex_path   = _TEX,
        columns        = _columns,
        column_headers = _column_headers,
        column_spec    = "ccl rr r r r",
        cell_format    = _cell_format,
        row_groups     = _row_groups,
        label          = f"tab:{_NAME}",
        caption        = TABLE_CAPTIONS.get(_NAME) or None,
        short_caption  = TABLE_CAPTIONS_SHORT.get(_NAME) or None,
    ),
)
print(f"   TEX → {_TEX}")


# %%
# [DATA: RENDER]  — analysis_scratch/plateau_values_table.py
"""
── CH04 § 4o (companion table) — Plateau A_FFT values inside chosen window ──
Per (f, amp, wind) cell: median A_IN, A_OUT, OUT/IN over the chosen window
(probe-shifted, N_off=7, N_len=10). 24 cells (4 freqs × 3 amps × 2 winds),
grouped by amp tier.
"""
_NAME = "ch04_plateau_values"
_CSV  = Path(f"output/TABLES/data/{_NAME}.csv")
_META = Path(f"output/TABLES/data/{_NAME}.meta.json")
_TEX  = Path(f"output/TABLES/{_NAME}.tex")


_AMP_TIERS = [
    (0.10, "A1", r"$A_1$"),
    (0.20, "A2", r"$A_2$"),
    (0.30, "A3", r"$A_3$"),
]
_WIND_LABEL = {"no": "uten", "full": "full"}


def _fmt_num_or_dash(value: float, decimals: int) -> str:
    if pd.isna(value):
        return r"\textendash"
    return rf"$\num{{{value:.{decimals}f}}}$"


_cell_format = {
    "freq":     lambda r: rf"$\num{{{r['freq']:.1f}}}$",
    "wind":     lambda r: _WIND_LABEL[r["wind"]],
    "n":        lambda r: rf"$\num{{{int(r['n'])}}}$",
    "A_in":     lambda r: _fmt_num_or_dash(r["A_in"],     2),
    "A_out":    lambda r: _fmt_num_or_dash(r["A_out"],    2),
    "Kt":       lambda r: _fmt_num_or_dash(r["Kt"],       3),
    "sigma_Kt": lambda r: _fmt_num_or_dash(r["sigma_Kt"], 3),
}

_columns        = ["freq", "wind", "n", "A_in", "A_out", "Kt", "sigma_Kt"]
_column_headers = [
    r"$f$ [\unit{\hertz}]",
    "vind",
    r"$n$",
    r"$A_\mathrm{Inn}$ [\unit{\milli\meter}]",
    r"$A_\mathrm{Ut}$ [\unit{\milli\meter}]",
    r"$K_t$",
    r"$\sigma (K_t)$",
]

_row_groups: list[tuple[str | None, callable]] = []
for _amp, _short, _label in _AMP_TIERS:
    _group_label = rf"\textbf{{{_label}}} ($V = {_amp:.2f}$ V)"
    _row_groups.append(
        (_group_label,
         (lambda a: lambda df: df[np.isclose(df["amp_v"], a)])(_amp))
    )

_render_with_caption_short(
    _META,
    TABLE_CAPTIONS_SHORT.get(_NAME) or "",
    lambda: render_table(
        csv_path       = _CSV,
        meta_path      = _META,
        out_tex_path   = _TEX,
        columns        = _columns,
        column_headers = _column_headers,
        column_spec    = "ccccccc",
        cell_format    = _cell_format,
        row_groups     = _row_groups,
        label          = f"tab:{_NAME}",
        caption        = TABLE_CAPTIONS.get(_NAME) or None,
        short_caption  = TABLE_CAPTIONS_SHORT.get(_NAME) or None,
    ),
)
print(f"   TEX → {_TEX}")


print("\nDone.")
