"""
Archived 2026-04-28: plot_damping_wind_delta (CH05 §3 wind-effect delta plot).

Replaced by `analysis_scratch/wind_effect_table.py`, which emits a LaTeX
table with both metrics (Δτ, % transmission gain, % damping reduction) per
(freq, amp) cell. The table is more thesis-friendly than the 2-row Δ scatter
this code produced — same data, different format. See
`output/TABLES/ch05_wind_effect_table.tex`.

Code below is the verbatim cut from `wavescripts/plotter.py` at archive time
(commit-staged 2026-04-28). Kept for reference; do NOT import.

Originally contained:
  - _make_damping_wind_delta_fig    (per-(panel, amp) 2-row figure builder)
  - _compute_damping_wind_delta_ylims (shared y-lims across voltage subfigs)
  - plot_damping_wind_delta          (public entry, called from main_save_figures.py CH05 §3)

Output files at archive time (also moved to ignore_this_archive/output/):
  - output/FIGURES/ch05_damping_wind_delta_full_A{1,2,3}.pdf
  - output/TEXFIGU/ch05_damping_wind_delta.tex
"""
# ═══════════════════════════════════════════════════════════════════════════════
# WIND DELTA
# ═══════════════════════════════════════════════════════════════════════════════


def _make_damping_wind_delta_fig(
    stats_df: pd.DataFrame,
    panel: str,
    amp: float,
    ref_wind: str = "no",
    target_wind: str = "full",
    figsize: tuple = (6, 5),
    ylim_top: Optional[Tuple[float, float]] = None,
    ylim_bot: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Two-row figure: top = OUT/IN per wind condition; bottom = delta (target − ref).
    Colour = wind condition (top only). Delta bar chart with sign-coded fill.
    Shared y-limits across voltage panels via ylim_top/ylim_bot overrides.
    """
    subset = stats_df[
        (stats_df[GC.PANEL_CONDITION] == panel)
        & (stats_df[GC.WAVE_AMPLITUDE_INPUT] == amp)
    ]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
    )

    # ── Top: OUT/IN per wind ──────────────────────────────────────────────────
    # Collapse multiple mooring groups at the same frequency into a single point
    # before plotting — otherwise the line zig-zags between mooring rows.
    for wind, grp in subset.groupby(GC.WIND_CONDITION):
        freq_agg = (
            grp.groupby(GC.WAVE_FREQUENCY_INPUT)
               .agg(mean_out_in=("mean_out_in", "mean"), std_out_in=("std_out_in", "mean"))
               .reset_index()
               .sort_values(GC.WAVE_FREQUENCY_INPUT)
        )
        ax_top.errorbar(
            freq_to_k(freq_agg[GC.WAVE_FREQUENCY_INPUT].values), freq_agg["mean_out_in"],
            yerr=freq_agg["std_out_in"].fillna(0),
            label=wind_to_label(wind), color=WIND_COLOR_MAP.get(wind, "gray"),
            marker="o", markersize=5, linewidth=1.4, capsize=3,
        )
    ax_top.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    ax_top.set_ylabel(r"$A_\mathrm{Ut}/A_\mathrm{inn}$", fontsize=9)
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(title="vind", fontsize=8, title_fontsize=8)
    add_freq_axis(ax_top)

    # ── Bottom: delta = target − ref ──────────────────────────────────────────
    # Aggregate per frequency first (multiple moorings → single mean per freq)
    ref_mean    = (subset[subset[GC.WIND_CONDITION] == ref_wind]
                   .groupby(GC.WAVE_FREQUENCY_INPUT)["mean_out_in"].mean())
    target_mean = (subset[subset[GC.WIND_CONDITION] == target_wind]
                   .groupby(GC.WAVE_FREQUENCY_INPUT)["mean_out_in"].mean())

    common_freqs = sorted(ref_mean.index.intersection(target_mean.index))
    if common_freqs:
        common_k = freq_to_k(np.array(common_freqs))
        delta = target_mean.loc[common_freqs] - ref_mean.loc[common_freqs]
        colors = [
            WIND_COLOR_MAP.get(target_wind, "steelblue") if d >= 0 else WIND_COLOR_MAP.get(ref_wind, "gray")
            for d in delta
        ]
        ax_bot.bar(common_k, delta.values, width=0.35, color=colors, alpha=0.75)
        ax_bot.axhline(0, color="black", linewidth=0.8)
        ax_bot.set_ylabel("Δ (full−no)", fontsize=8)
    else:
        ax_bot.text(0.5, 0.5, f"No matched ({ref_wind}/{target_wind}) points",
                    ha="center", va="center", transform=ax_bot.transAxes, fontsize=9, color="gray")

    ax_bot.set_xlabel("$k$ (rad/m)", fontsize=9)
    ax_bot.grid(True, alpha=0.3)

    if ylim_top is not None:
        ax_top.set_ylim(ylim_top)
    if ylim_bot is not None:
        ax_bot.set_ylim(ylim_bot)

    fig.subplots_adjust(left=0.14, right=0.97, top=0.83, bottom=0.11)
    return fig


def _compute_damping_wind_delta_ylims(
    stats_df: pd.DataFrame,
    panel_conditions: list,
    amplitudes: list,
    ref_wind: str,
    target_wind: str,
    pad_frac: float = 0.05,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute shared y-limits for the top (OUT/IN) and bottom (Δ) panels
    across every (panel × amp) combination, so the per-voltage PDFs use
    identical axes for direct visual comparison.
    """
    top_lo, top_hi = np.inf, -np.inf
    bot_lo, bot_hi = np.inf, -np.inf
    for panel in panel_conditions:
        for amp in amplitudes:
            sub = stats_df[
                (stats_df[GC.PANEL_CONDITION] == panel)
                & (stats_df[GC.WAVE_AMPLITUDE_INPUT] == amp)
            ]
            if sub.empty:
                continue
            std = sub.get("std_out_in", pd.Series(0, index=sub.index)).fillna(0)
            y = sub["mean_out_in"].astype(float)
            top_lo = min(top_lo, float((y - std).min()))
            top_hi = max(top_hi, float((y + std).max()))

            ref_mean = (sub[sub[GC.WIND_CONDITION] == ref_wind]
                        .groupby(GC.WAVE_FREQUENCY_INPUT)["mean_out_in"].mean())
            tgt_mean = (sub[sub[GC.WIND_CONDITION] == target_wind]
                        .groupby(GC.WAVE_FREQUENCY_INPUT)["mean_out_in"].mean())
            common = ref_mean.index.intersection(tgt_mean.index)
            if len(common):
                delta = (tgt_mean.loc[common] - ref_mean.loc[common]).astype(float)
                bot_lo = min(bot_lo, float(delta.min()))
                bot_hi = max(bot_hi, float(delta.max()))

    if not np.isfinite(top_lo):
        top_lo, top_hi = 0.0, 1.2
    if not np.isfinite(bot_lo):
        bot_lo, bot_hi = -0.2, 0.2

    top_pad = (top_hi - top_lo) * pad_frac or 0.02
    bot_span = max(abs(bot_lo), abs(bot_hi))
    bot_pad = bot_span * pad_frac or 0.02
    bot_sym = bot_span + bot_pad
    return (top_lo - top_pad, top_hi + top_pad), (-bot_sym, bot_sym)


def plot_damping_wind_delta(
    stats_df: pd.DataFrame,
    plotvariables: Optional[dict] = None,
    chapter: str = "05",
) -> None:
    """
    Wind-effect delta plot: OUT/IN(fullwind) − OUT/IN(nowind) vs frequency.
    One two-row figure per (panel × amplitude): top = raw OUT/IN per wind,
    bottom = signed delta bar chart.

    show_plot → one window per combination (REPL)
    save_plot → one PDF per combination + .tex stub

    Input: output from damping_all_amplitude_grouper()
    """
    if plotvariables is None:
        plotvariables = {"plotting": {"show_plot": True, "save_plot": False}}

    plotting  = plotvariables.get("plotting", {})
    show_plot = plotting.get("show_plot", False)
    save_plot = plotting.get("save_plot", False)
    figsize   = plotting.get("figsize", (6, 5))
    ref_wind    = plotting.get("ref_wind",    "no")
    target_wind = plotting.get("target_wind", "full")

    panel_conditions = sorted(stats_df[GC.PANEL_CONDITION].unique())
    amplitudes       = sorted(stats_df[GC.WAVE_AMPLITUDE_INPUT].unique())
    wind_conditions  = sorted(stats_df[GC.WIND_CONDITION].unique())
    n_runs           = int(stats_df["n_runs"].sum()) if "n_runs" in stats_df.columns else len(stats_df)

    ylim_top, ylim_bot = _compute_damping_wind_delta_ylims(
        stats_df, panel_conditions, amplitudes, ref_wind, target_wind,
    )

    _top_caption = plotvariables.get("caption")
    if isinstance(_top_caption, str) and "caption" not in plotting:
        plotting = {**plotting, "caption": _top_caption}

    _caption_slots = {
        "n_runs":     n_runs,
        "n_panels":   len(panel_conditions),
        "panels":     ", ".join(panel_conditions),
        "ref_wind":   ref_wind,
        "target_wind": target_wind,
        "amps":       ", ".join(amp_to_label(a) for a in amplitudes),
    }
    _default_caption = ""   # caption text comes from FIGURE_CAPTIONS dict
    _caption = resolve_caption(plotting, _default_caption, _caption_slots,
                               fn_name="plot_damping_wind_delta")

    if show_plot:
        for panel in panel_conditions:
            for amp in amplitudes:
                fig = _make_damping_wind_delta_fig(
                    stats_df, panel, amp, ref_wind, target_wind, figsize=figsize,
                    ylim_top=ylim_top, ylim_bot=ylim_bot,
                )
                plt.show()

    if save_plot:
        subfig_filenames = []
        # Headline wind-effect numbers per amplitude (delta = target − ref,
        # averaged across frequencies within each amplitude).
        _extra_stats = {
            "n_panels":     len(panel_conditions),
            "n_amplitudes": len(amplitudes),
            "ref_wind":     ref_wind,
            "target_wind":  target_wind,
        }
        for amp in amplitudes:
            _sub_amp = stats_df[
                (stats_df[GC.WAVE_AMPLITUDE_INPUT] == amp)
                & (stats_df[GC.PANEL_CONDITION].isin(panel_conditions))
            ]
            _ref = (_sub_amp[_sub_amp[GC.WIND_CONDITION] == ref_wind]
                    .groupby(GC.WAVE_FREQUENCY_INPUT)["mean_out_in"].mean())
            _tgt = (_sub_amp[_sub_amp[GC.WIND_CONDITION] == target_wind]
                    .groupby(GC.WAVE_FREQUENCY_INPUT)["mean_out_in"].mean())
            _common = _ref.index.intersection(_tgt.index)
            if len(_common):
                _delta = (_tgt.loc[_common] - _ref.loc[_common]).astype(float)
                _amp_tag = amp_to_tag(amp)
                _extra_stats[f"mean_delta_{_amp_tag}"] = round(float(_delta.mean()), 4)
                _extra_stats[f"max_abs_delta_{_amp_tag}"] = round(float(_delta.abs().max()), 4)

        meta_base = build_fig_meta(
            {**plotvariables, "plotting": {**plotting, "caption": _caption}},
            chapter=chapter,
            extra={"script": "plotter.py::plot_damping_wind_delta"},
            data_df=stats_df,
            computed_in="filters.py::damping_all_amplitude_grouper → plotter.py::_make_damping_wind_delta_fig",
            data_class="META",
            findings_doc=None,
            grouper="damping_all_amplitude_grouper",
            collapse_panels=False,
            fft_window_hz=0.1,
            extra_params=(
                f"window=0.1 Hz, ref_wind={ref_wind}, target_wind={target_wind}, "
                "delta=mean_out_in(target)-mean_out_in(ref) aggregated per frequency, "
                "shared y-lims across voltage subfigures"
            ),
            extra_stats=_extra_stats,
        )
        figure_name = plotting.get("figure_name") or build_filename("damping_wind_delta", meta_base)
        subfig_captions = []
        for panel in panel_conditions:
            for amp in amplitudes:
                fig_s = _make_damping_wind_delta_fig(
                    stats_df, panel, amp, ref_wind, target_wind, figsize=figsize,
                    ylim_top=ylim_top, ylim_bot=ylim_bot,
                )
                amp_tag = amp_to_tag(amp)
                fname = f"{figure_name}_{panel}_{amp_tag}"
                _save_figure(fig_s, fname, save_pgf=True)
                subfig_filenames.append(fname)
                subfig_captions.append(f"{panel.capitalize()} panel, {amp_to_label(amp)}")
                plt.close(fig_s)

        stub_meta = {**meta_base, "panel": panel_conditions, "wind": f"{ref_wind}_vs_{target_wind}"}
        write_figure_stub(stub_meta, "damping_wind_delta", subfig_filenames=subfig_filenames,
                          subfig_captions=subfig_captions,
                          force=plotting.get("force_stub", True))

