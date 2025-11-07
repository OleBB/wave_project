#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 21:16:15 2025

@author: gpt uio
"""

#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime

PARENT = Path(r"/Users/ole/Kodevik/wave_project/pressuredata")

def main():
    if not PARENT.is_dir():
        raise SystemExit(f"Not a directory: {PARENT}")

    folder_names = sorted([p.name for p in PARENT.iterdir() if p.is_dir()])

    data = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "parent": str(PARENT),
        "folders": folder_names,
        "config": {
            "pattern": r"moh(\d{3})",
            "file_ext": ".txt",
            "flip_axes": False
        },
        "notes": ""
    }

    out_path = PARENT / "selected_folders.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()

"""#!/usr/bin/env python3 #GPT: EXAMPLE “one step up, two steps down” anchored to the script’s directory
from pathlib import Path

# Directory where this script lives
HERE = Path(__file__).resolve().parent

# Go 1 step up, then into "data/FolderA"
target = HERE.parent / "data" / "FolderA"

print(target)
# Now safely use target regardless of where you run the script:
# for p in target.iterdir(): ...
"""

"""from pathlib import Path
##If you want to be project-root anchored:
# Suppose your project root has a marker file (e.g., .git or pyproject.toml)
ROOT = Path(__file__).resolve().parents[1]  # adjust depth or detect via marker
DATA = ROOT / "data"
folderA = DATA / "FolderA"
"""