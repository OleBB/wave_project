# -*- coding: utf-8 -*-


# ------------------------------------------------------------
# Core plot function (single dataset)
# ------------------------------------------------------------
# def plot_column(df, start, end, chosenprobe, title="", ax=None,
#                 color="black", linestyle="-"):

#     if ax is None:
#         fig, ax = plt.subplots(figsize=(10, 5))
    
#     ax.plot(df[chosenprobe].iloc[start:end],
#             label=title,
#             color=color,
#             linestyle=linestyle)

#     ax.set_title(title)
#     return ax
# ------------------------------------------------------------
# ... handles the input from main and runs plot_filtered
# ... Choose to plot separate plots or a combined overlaid plot
# ------------------------------------------------------------
# def plotter_selection(processed_dfs, df_sel, plotvariables):

#     manual_start = plotvariables["processing"]["rangestart"]
#     manual_end   = plotvariables["processing"]["rangeend"]
    
#     """Estimated Probe 1 start"""
    
#     """... tenker å lage en knapp som velger 
#     av eller på manuell range vs calculated range 
#     den dataaen må jo lagres til meta.json"""
#     # debug - print types and a small sample
#     print("type(df_sel) =", type(df_sel))
#     try:
#         print("df_sel sample (first 5):", list(df_sel)[:5])
#     except Exception:
#         print("Could not list df_sel")
#     #print("type(auto_ranges) =", type(auto_ranges))
#     #print("auto_ranges keys (first 10):", list(auto_ranges.keys())[:10])

#     # ---- compute plot ranges per path ----
#     plot_ranges = {}
    
#     for path in df_sel["path"]: #pleide å være processed_dfs
#         #auto_start, auto_end = auto_ranges[path]
#         start = manual_start if manual_start is not None else None#siste her pleide å være auto
#         end   = manual_end   if manual_end   is not None else None
#         plot_ranges[path] = (start, end)
    
#     print('plot ranges variable:',plot_ranges)    
#     # ---- RUN SEPARATE PLOTS ----
#     if plotvariables["plotting"]["separate"]:
#         for path, df_ma in processed_dfs.items():

#             plot_start, plot_end = plot_ranges[path]

#             runtime_vars = {
#                 **plotvariables["processing"],
#                 **plotvariables["plotting"],
#                 "rangestart": plot_start,
#                 "rangeend": plot_end,
#             }

#             plot_filtered(
#                 processed_dfs={path: df_ma},
#                 df_sel=df_sel[df_sel["path"] == path],
#                 **runtime_vars
#             )

#     # ---- RUN OVERLAYED PLOT ----
#     if  plotvariables["plotting"]["overlay"]:
#         plot_overlayed(
#             processed_dfs,
#             df_sel,
#             plot_ranges,    # <-- instead of auto_ranges
#             plotvariables
#         )

# ------------------------------------------------------------
# old function: filters metadata, smooths, colors, styles, plots
# ------------------------------------------------------------
# def plot_filtered(processed_dfs,
#                   df_sel,
#                   **plotvariables):
#     #unpack plotvariables/kwargs:
#     chosenprobe = plotvariables["chosenprobe"]
#     rangestart  = plotvariables["rangestart"]
#     rangeend    = plotvariables["rangeend"]
#     win         = plotvariables["win"]
#     figsize     = plotvariables.get("figsize")

#     plotting = plotvariables.get("plotting", {})
#     show_plot = plotting.get("show_plot", True)
#     save_plot = plotting.get("save_plot", False)
    
#     chosenprobe = plotvariables["processing"]["chosenprobe"]
#     figsize     = plotting.get("figsize", (10, 6))
    
#     fig, ax = plt.subplots(figsize=figsize)
    
#     chapter = 5

#     for idx, row in df_sel.iterrows():
        
#         path_key = row["path"]
#         df_ma   = processed_dfs[path_key]
#         print("Columns for", row["path"], df_ma.columns.tolist())

#         # Color based on wind
#         windcond = row["WindCondition"]
#         color = WIND_COLOR_MAP.get(windcond, "black")

#         # Linestyle based on panel condition
#         # panelcond = row["PanelCondition"] -delete?
#         # linestyle = "--" if "full" in panelcond else "-" -delete?
#         lstyle   = PANEL_STYLES.get(row.get("PanelCondition", ""), "solid")


#         # Short label for legend
#         label = _make_label(row)
        
#         print("start_idx =", rangestart)
#         print("end_idx   =", rangeend)
#         print("df len    =", len(df_ma))
#         df_cut = df_ma.loc[rangestart:rangeend]
#         print("df_cut len:", len(df_cut))
#         #print(df_cut["Date"].head())
        
#         # Convert Date column to milliseconds relative to the start
#         t0 = df_cut["Date"].iloc[0]
#         time_ms = (df_cut["Date"] - t0).dt.total_seconds() * 1000
    
#         # NEW — automatically uses zeroed signal if it exists
#         probe_num = chosenprobe.split()[-1] # extracts "1" from "Probe 1"
#         zeroed_col = f"eta_{probe_num}"
        
#         if zeroed_col in df_cut.columns:
#             y_data = df_cut[zeroed_col]
#             ylabel = f"{zeroed_col} (zeroed)"
#         else:
#             y_data = df_cut[chosenprobe]
#             ylabel = chosenprobe

#         ax.plot(time_ms, y_data, label=label, color=color, linestyle=lstyle)
#         ax.set_ylabel(ylabel)

#     ax.set_xlabel("Milliseconds")
#     ax.set_ylabel(chosenprobe)
#     ax.set_title(f"{chosenprobe} — smoothed (window={win})")
#     ax.legend()

#     # ── Save ─────────────────────────────────────────────────────────────────
#     if save_plot:
#         meta = build_fig_meta(plotvariables, chapter=chapter,
#                               extra={"script": "plotter.py::plot_timeseries"})
#         save_and_stub(fig, meta, plot_type="timeseries",
#                       save_pgf=plotting.get("save_pgf", True))

#     if show_plot:
#         plt.show()
#     else:
#         plt.close(fig)

# def plot_overlayed(processed_dfs, df_sel, plot_ranges, plotvariables):
#     """
#     Overlay multiple datasets on the same axes,
#     aligning each on its own good_start_idx, and using
#     the same legend style as plot_filtered (_make_label).
#     """

#     plotting = plotvariables.get("plotting", {})
#     show_plot = plotting.get("show_plot", True)
#     save_plot = plotting.get("save_plot", False)

#     chosenprobe = plotvariables["processing"]["chosenprobe"]
#     figsize     = plotting.get("figsize", (10, 6))

#     fig, ax = plt.subplots(figsize=figsize)
    
#     chapter = 5

#     for idx, row in df_sel.iterrows():
#         path = row["path"]
        
#         # Color based on wind
#         windcond = row["WindCondition"]
#         color = WIND_COLOR_MAP.get(windcond, "black")
        
#         # skip if this path isn't in processed_dfs/auto_ranges
#         if path not in processed_dfs or path not in plot_ranges:
#             continue

#         df_ma = processed_dfs[path]
#         start_idx, end_idx = plot_ranges[path]

#         # slice the good part
#         df_cut = df_ma.iloc[start_idx:end_idx]

#         if df_cut.empty:
#             continue

#         # time in ms relative to local start
#         t0 = df_cut["Date"].iloc[0]
#         time_ms = (df_cut["Date"] - t0).dt.total_seconds() * 1000

#         # use your existing label function
#         label = _make_label(row)

#         ax.plot(time_ms,
#                 df_cut[chosenprobe],
#                 label=label,
#                 color=color)

#     ax.set_xlabel("Time [ms, aligned at clean start]")
#     ax.set_ylabel(chosenprobe)
#     ax.set_title(f"{chosenprobe} — overlayed clean segments")
#     ax.legend()
#     plt.tight_layout()
#     # ── Save ─────────────────────────────────────────────────────────────────
#     if save_plot:
#         meta = build_fig_meta(plotvariables, chapter=chapter,
#                               extra={"script": "plotter.py::plot_timeseries"})
#         save_and_stub(fig, meta, plot_type="timeseries",
#                       save_pgf=plotting.get("save_pgf", True))

#     if show_plot:
#         plt.show()
#     else:
#         plt.close(fig)




# def old_plot_ramp_detection(df, meta_sel, data_col,
#                         signal,
#                         baseline_mean,
#                         threshold,
#                         first_motion_idx,
#                         good_start_idx,
#                         good_range,
#                         good_end_idx,
#                         peaks=None,
#                         peak_amplitudes=None,
#                         ramp_peak_indices=None,
#                         title="Ramp Detection Debug"):
#     t0 = df["Date"].iat[0]
#     time = (df["Date"]-t0).dt.total_seconds() *1000
#     raw = df[data_col].values #bør jeg sette minus for å flippe hele greien?

#     plt.figure(figsize=(15, 7))
    
    
#     amp = meta_sel["WaveAmplitudeInput [Volt]"]

#     time_cut = time[good_start_idx:good_end_idx]
#     # print(time_cut)
#     sinewave = 100*amp*np.sin(0.004*1.3*time_cut)+baseline_mean
    
#     # 1. Plot raw + smoothed + sine
#     plt.plot(time, raw, color="lightgray", alpha=0.6, label="Raw signal")
#     plt.plot(time, signal, color="black", linewidth=2, label=f"Smoothed {data_col}")
#     # plt.plot(time_cut, sinewave, color="red", linestyle="--")

#     # 2. Baseline & threshold
#     plt.axhline(baseline_mean, color="blue", linestyle="--", label=f"Baseline = {baseline_mean:.2f} mm")
#     plt.axhline(baseline_mean + threshold, color="red", linestyle=":", alpha=0.7)
#     plt.axhline(baseline_mean - threshold, color="red", linestyle=":", alpha=0.7)

#     # 3. First motion
#     plt.axvline(time[first_motion_idx], color="orange", linewidth=2, linestyle="--",
#                 label=f"First motion #{first_motion_idx}")

#     # 4. Good stable interval
#     good_start_idx = int(good_start_idx)
#     good_end_idx = min(len(time) - 1, good_start_idx + int(good_range) - 1)

#     plt.axvline(time[good_start_idx], color="green", linewidth=3, label=f"Stable start #{good_start_idx}")
#     plt.axvline(time[good_end_idx], color="purple", linewidth=2, linestyle="--", label=f"End #{good_end_idx}")
#     plt.axvspan(time[good_start_idx], time[good_end_idx], color="green", alpha=0.15, label="Stable region")

#     # 5. Optional: highlight detected peaks and ramp-up
#     if peaks is not None:
#         plt.plot(time[peaks], signal[peaks], "ro", markersize=6, alpha=0.7, label="Detected peaks")
#     if ramp_peak_indices is not None:
#         plt.plot(time[ramp_peak_indices], signal[ramp_peak_indices],
#                  "o", color="lime", markersize=10, markeredgecolor="darkgreen", markeredgewidth=2,
#                  label=f"Ramp-up ({len(ramp_peak_indices)} peaks)")

#     # ────────────────── MAGIC ZOOM THAT MAKES THE WAVE VISIBLE ──────────────────
#     zoom_margin = 15  # mm above/below baseline — perfect for your ±8 mm waves
#     plt.ylim(baseline_mean - zoom_margin, baseline_mean + zoom_margin)
#     # ─────────────────────────────────────────────────────────────────────────────

#     # Title from filename
#     # New – works whether metadataframe is DataFrame or single row (Series)
#     path_value = meta_sel["path"] if isinstance(meta_sel, pd.Series) else meta_sel["path"].iloc[0]
#     filename = str(path_value).split("/")[-1]
#     plt.title(f"{filename}  →  {data_col}", fontsize=14, pad=20)

#     plt.xlabel("Time [ms]")
#     plt.ylabel("Water level [mm]")
#     plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
#     plt.grid(True, alpha=0.1)
#     plt.tight_layout()
    
# def old_plot_p2_vs_p3_scatter(band_amplitudes):
#     band_name = ['Swell', 'Wind', 'Total']
#     fig, axes = plt.subplots(1, len(band_name), figsize=(12, 4), sharex=False, sharey=False)
#     band_constants = {
#         'Swell': PC.SWELL_AMPLITUDE_PSD,
#         'Wind': PC.WIND_AMPLITUDE_PSD,
#         'Total': PC.TOTAL_AMPLITUDE_PSD,
#     }
    
#     for ax, (band_name, constant_template) in zip(axes, band_constants.items()):
#         p2_in = band_amplitudes[constant_template.format(i=2)].to_numpy()
#         p3_in = band_amplitudes[constant_template.format(i=3)].to_numpy()
#         # print(p2)
#         # print(p3)
#         # CREATE VALID MASK - remove NaN and inf
#         valid_mask = np.isfinite(p2_in) & np.isfinite(p3_in)
#         p2 = p2_in[valid_mask]
#         p3 = p3_in[valid_mask]
        
#         ax.scatter(p2, p3, alpha=0.7)
        
#         # Calculate limits
#         lim = max(p2.max(), p3.max()) * 1.05 if len(p2) else 1.0

#         # Plot reference line FIRST (or use zorder)
#         ax.plot([0, lim], [0, lim], 'k--', linewidth=1, label='y=x', zorder=1)
        
#         # Set axis limits to show the reference line
#         ax.set_xlim(0, lim)
#         ax.set_ylim(0, lim)
        
#         ax.set_title(f'{band_name}')
#         ax.set_xlabel('P2 amplitude')
#         ax.set_ylabel('P3 amplitude')
#         ax.grid(True, alpha=0.3)
#         ax.set_aspect('equal')  # Optional: makes it a square plot
    
#     plt.tight_layout()
#     plt.show()
    