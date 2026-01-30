#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 16:52:22 2025

@author: ole
"""

#!/usr/bin/env python3
# markdown report helper for wave-tank analysis
import os
import textwrap
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tabulate

def _safe_col(df, col):
    return df[col] if col in df.columns else pd.Series([np.nan]*len(df), index=df.index)

def markdown_report_from_df(df,
                            title="Wave Tank Analysis Report",
                            subtitle=None,
                            out_path="report.md",
                            plots_folder="reportplots",
                            save_plots=True,
                            max_rows=50):
    """
    Produce a simple Markdown report from a results DataFrame.

    Parameters
    - df : pandas.DataFrame  (one row per run)
    - title : str
    - subtitle : str or None
    - out_path : output markdown file path
    - plots_folder : where png previews are saved (if save_plots True)
    - save_plots : boolean, create small preview plots for each file/run (requires original data available in df['file_path'] or df['path'])
    - max_rows : how many rows to include in the summary table (keeps report readable)
    """
    out_path = Path(out_path)
    plots_folder = Path(plots_folder)
    plots_folder.mkdir(parents=True, exist_ok=True)

    # Select columns we like to show (if available)
    cols_want = [
        ("file", "File"),
        ("amp1", "P1"),
        ("amp2", "P2"),
        ("amp3", "P3"),
        ("amp4", "P4"),
        ("r21", "P2/P1"),
        ("r32", "P3/P2"),
        ("r43", "P4/P3"),
        ("lag12_ms", "Lag12 (ms)"),
        ("lag23_ms", "Lag23 (ms)"),
        ("celerity_m_s", "Celerity (m/s)"),
        ("verdict", "Verdict"),
    ]

    # Build table header (only include columns that exist in df)
    present = [(k,h) for k,h in cols_want if k in df.columns]
    if not present:
        # If nothing matches, just save the raw dataframe as markdown
        md_table = df.head(max_rows).to_markdown(index=False)
    else:
        headers = [h for k,h in present]
        keys = [k for k,h in present]
        # Format numeric columns nicely
        rows = []
        for i, row in df.head(max_rows).iterrows():
            vals = []
            for k in keys:
                v = row.get(k, None)
                if pd.isna(v):
                    vals.append("—")
                elif isinstance(v, (float, np.floating)):
                    # choose formatting by magnitude
                    if abs(v) >= 100:
                        vals.append(f"{v:.0f}")
                    elif abs(v) >= 1:
                        vals.append(f"{v:.2f}")
                    else:
                        vals.append(f"{v:.3f}")
                else:
                    vals.append(str(v))
            rows.append(vals)
        # create a markdown table
        col_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join("---" for _ in headers) + " |"
        data_lines = ["| " + " | ".join(r) + " |" for r in rows]
        md_table = "\n".join([col_line, sep_line] + data_lines)

    # Build top metadata block
    md_lines = []
    md_lines.append(f"# {title}")
    if subtitle:
        md_lines.append(f"**{subtitle}**")
    md_lines.append("")
    md_lines.append(f"_Generated: {pd.Timestamp.now()}_")
    md_lines.append("")
    md_lines.append("## Summary")
    md_lines.append("")
    # basic summary stats for damping ratios if present
    if "r32" in df.columns:
        valid = df["r32"].dropna().astype(float)
        if len(valid):
            md_lines.append(f"- P3/P2 over runs: mean = {valid.mean():.3f}, std = {valid.std():.3f}, n = {len(valid)}")
    if "r21" in df.columns:
        valid = df["r21"].dropna().astype(float)
        if len(valid):
            md_lines.append(f"- P2/P1 over runs: mean = {valid.mean():.3f}, std = {valid.std():.3f}, n = {len(valid)}")
    md_lines.append("")
    md_lines.append("## Detailed results (first rows)")
    md_lines.append("")
    md_lines.append(md_table)
    md_lines.append("")

    # Optionally add small plots for first N rows if original path available
    if save_plots and ("file" in df.columns or "path" in df.columns or "file_path" in df.columns):
        path_col = "file" if "file" in df.columns else ("file_path" if "file_path" in df.columns else "path")
        md_lines.append("## Preview plots")
        md_lines.append("")
        # User must supply the processed_dfs mapping externally if you want to auto-plot time series.
        # Here we only include existing plot files if present; otherwise try to create generic placeholders.
        for i, row in df.head(max_rows).iterrows():
            p = row.get(path_col, None)
            if not p:
                continue
            fname = Path(p).name
            plot_name = plots_folder / f"{Path(fname).stem}_preview.png"
            # If plot file doesn't exist, try to create a tiny placeholder (safe)
            if not plot_name.exists():
                # create a small placeholder image indicating "no preview"
                fig, ax = plt.subplots(figsize=(4,2))
                ax.text(0.5, 0.5, "preview not generated", ha="center", va="center", fontsize=8)
                ax.set_axis_off()
                fig.tight_layout()
                fig.savefig(plot_name, dpi=150)
                plt.close(fig)
            md_lines.append(f"### {fname}")
            md_lines.append(f"![preview]({plot_name.as_posix()})")
            md_lines.append("")
    # Footer
    md_lines.append("---")
    md_lines.append("Report produced by wave-tank analysis helper.")
    md_content = "\n".join(md_lines)

    # Write file
    out_path.write_text(md_content, encoding="utf8")
    print(f"Markdown report written to: {out_path.resolve()}")
    return out_path

# Example minimal usage:
if __name__ == "__main__":
    # suppose `results` is the DataFrame your diagnostic functions return:
    # results = full_tank_diagnostics(processed_dfs)
    # markdown_report_from_df(results, title="Full Tank Diagnostics", out_path="diagnostics.md")
    pass
