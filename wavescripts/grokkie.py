# ——— FINAL PRODUCTION VERSION ———
from pathlib import Path
from typing import Iterator, Dict, Tuple
import pandas as pd

def get_data_files(folder: Path) -> Iterator[Path]:
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        return
    patterns = ["*.csv", "*.CSV", "*.parquet", "*.h5", "*.feather"]
    total = 0
    for pat in patterns:
        matches = list(folder.rglob(pat))
        if matches:
            print(f"  Found {len(matches)} files with {pat}")
            total += len(matches)
            yield from matches
    if total == 0:
        print(f"  No files in {folder}")

def load_or_update(
    cache_dir: Path = Path("data_cache"),
    *folders: Path
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load cached DataFrames + add new files from folders.
    Creates cache_dir if missing.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dfs_path = cache_dir / "dfs.pkl"
    meta_path = cache_dir / "meta.parquet"

    # --- Load existing cache ---
    dfs: Dict[str, pd.DataFrame] = {}
    meta = pd.DataFrame()
    if dfs_path.exists() and meta_path.exists():
        print("Loading cached data...")
        try:
            dfs = pd.read_pickle(dfs_path)
            meta = pd.read_parquet(meta_path)
            print(f" → {len(dfs)} files already cached")
        except Exception as e:
            print(f"Cache corrupted ({e}), rebuilding...")
    else:
        print("No cache found → starting fresh")

    # --- Find new files ---
    seen = set(dfs.keys())
    new_files = []
    for folder in folders:
        folder = Path(folder)
        if not folder.is_dir():
            print(f"Warning: Folder not found: {folder}")
            continue
        for path in get_data_files(folder):
            if path.name not in seen:
                new_files.append(path)

    if not new_files:
        print("No new files found.")
        return dfs, meta

    print(f"Loading {len(new_files)} new files...")
    # ←←← PUT YOUR ACTUAL PARSING CODE HERE →→→
    for i, path in enumerate(new_files, 1):
        try:
            # Example: adjust based on your file type
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
            elif path.suffix == ".h5":
                df = pd.read_hdf(path)
            elif path.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                print(f"  Skipping unknown type: {path}")
                continue

            dfs[path.name] = df
            # Example metadata extraction:
            row = {
                "filename": path.name,
                "path": str(path),
                "rows": len(df),
                "cols": len(df.columns),
                "size_mb": path.stat().st_size / 1e6
            }
            meta = pd.concat([meta, pd.DataFrame([row])], ignore_index=True)
            print(f"  [{i}/{len(new_files)}] Loaded {path.name} → {len(df):,} rows")

        except Exception as e:
            print(f"  Failed to load {path.name}: {e}")

    # --- Save updated cache ---
    pd.to_pickle(dfs, dfs_path)
    meta.to_parquet(meta_path, index=False)
    print(f"Cache updated → {len(dfs)} total files in {cache_dir}")

    return dfs, meta


# ——— RUN IT ———
FOLDER1 = Path("/Users/ole/Kodevik/wave_project/wavedata/20251110-tett6roof-lowM-ekte580")
print(f"CALLING load_or_update with: {FOLDER1}")
dfs, meta = load_or_update(FOLDER1)


print(f"\nSUCCESS: {len(dfs)} DataFrames loaded and cached!")
print(f"Metadata shape: {meta.shape}")