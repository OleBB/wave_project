"""
REFACTORING GUIDE: How to update your data_loader.py

This shows the BEFORE and AFTER for the key sections.
"""

# =============================================================================
# STEP 1: Add to constants.py
# =============================================================================

# Add this to your wavescripts/constants.py:

"""
@dataclass(frozen=True)
class MeasurementConstants:
    SAMPLING_RATE: float = 250.0
    NUM_PROBES: int = 4
    PROBE_NAMES: tuple = ("Probe 1", "Probe 2", "Probe 3", "Probe 4")
    MM_TO_M: float = 0.001
    M_TO_MM: float = 1000.0
    
    # NEW: Stillwater calculation
    STILLWATER_SAMPLES: int = 500  # Number of samples to use for stillwater (2 seconds)
"""


# =============================================================================
# STEP 2: Replace the probe position logic in data_loader.py
# =============================================================================

# ────────────────────────────────────────────────────────────
# BEFORE (your current code around line 180)
# ────────────────────────────────────────────────────────────

"""
                #Probe distance logic
                # "Probe 1 mm from paddle"
                modtime = os.path.getmtime(path)
                file_date = datetime.fromtimestamp(modtime)
                date_match = re.search(r'(\d{8})', filename)
                distance_cutoff = datetime(2025, 11, 14) #siste kjøring var 13nov
                metadata["Probe 1 mm from paddle"] = 8855 if file_date < distance_cutoff else None
                metadata["Probe 2 mm from paddle"] = 9455 if file_date < distance_cutoff else None
                metadata["Probe 3 mm from paddle"] = 12544 if file_date < distance_cutoff else None
                metadata["Probe 4 mm from paddle"] = 12545 if file_date < distance_cutoff else None
"""

# ────────────────────────────────────────────────────────────
# AFTER (new configuration-based code)
# ────────────────────────────────────────────────────────────

"""
                # Probe distance logic (using configuration system)
                file_date = _extract_file_date(filename, path)
                if file_date:
                    try:
                        probe_positions = get_probe_positions(file_date)
                        for probe_num, distance in probe_positions.items():
                            metadata[f"Probe {probe_num} mm from paddle"] = distance
                    except ValueError as e:
                        print(f"   Warning: No probe config for {path.name}: {e}")
                        for probe_num in range(1, 5):
                            metadata[f"Probe {probe_num} mm from paddle"] = None
                else:
                    for probe_num in range(1, 5):
                        metadata[f"Probe {probe_num} mm from paddle"] = None
"""


# =============================================================================
# STEP 3: Do the same for mooring logic
# =============================================================================

# ────────────────────────────────────────────────────────────
# BEFORE (your current code around line 165)
# ────────────────────────────────────────────────────────────

"""
                # Mooring logic
                modtime = os.path.getmtime(path)
                file_date = datetime.fromtimestamp(modtime)
                date_match = re.search(r'(\d{8})', filename)
                mooring_cutoff = datetime(2025, 11, 6)
                metadata["Mooring"] = "high" if file_date < mooring_cutoff else "low"
                if date_match:
                    metadata["Date"] = date_match.group(1)
                    if metadata["Date"] < "20251106":
                        metadata["Mooring"] = "high"
                    else:
                        metadata["Mooring"] = "low"
"""

# ────────────────────────────────────────────────────────────
# AFTER (new configuration-based code)
# ────────────────────────────────────────────────────────────

"""
                # Mooring logic (using configuration system)
                file_date = _extract_file_date(filename, path)
                if file_date:
                    metadata["Mooring"] = get_mooring_type(file_date)
                else:
                    metadata["Mooring"] = "unknown"
"""


# =============================================================================
# STEP 4: Add these helper functions at top of data_loader.py
# =============================================================================

"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProbeConfiguration:
    '''A set of probe positions valid for a date range.'''
    name: str
    valid_from: datetime
    valid_until: Optional[datetime]
    distances_mm: Dict[int, float]
    notes: str = ""


# Define all configurations
PROBE_CONFIGS = [
    ProbeConfiguration(
        name="initial_setup",
        valid_from=datetime(2025, 1, 1),
        valid_until=datetime(2025, 11, 14),
        distances_mm={1: 8855.0, 2: 9455.0, 3: 12544.0, 4: 12545.0},
        notes="Original configuration until Nov 13, 2025"
    ),
    ProbeConfiguration(
        name="nov14_adjustment",
        valid_from=datetime(2025, 11, 14),
        valid_until=None,
        distances_mm={1: 8855.0, 2: 9455.0, 3: 12544.0, 4: 12545.0},  # TODO: Update!
        notes="Configuration after Nov 14. TODO: Get actual values!"
    ),
]


def get_probe_positions(file_date: datetime) -> Dict[int, float]:
    '''Get probe positions for a given date.'''
    for config in PROBE_CONFIGS:
        if config.valid_from <= file_date:
            if config.valid_until is None or file_date < config.valid_until:
                return config.distances_mm.copy()
    raise ValueError(f"No probe configuration found for date {file_date}")


def _extract_file_date(filename: str, file_path: Path) -> Optional[datetime]:
    '''Extract date from filename, fall back to modification time.'''
    # Try filename first
    date_match = re.search(r'(\\d{8})', filename)
    if date_match:
        try:
            return datetime.strptime(date_match.group(1), "%Y%m%d")
        except ValueError:
            pass
    
    # Fall back to file modification time
    try:
        modtime = os.path.getmtime(file_path)
        return datetime.fromtimestamp(modtime)
    except Exception:
        return None
"""


# =============================================================================
# STEP 5: When you change probe positions in the future
# =============================================================================

"""
LET'S SAY: On January 20, 2026, you move the probes to new positions.

HERE'S ALL YOU DO:

1. Update PROBE_CONFIGS list:

   # Change the previous config's end date:
   ProbeConfiguration(
       name="nov14_adjustment",
       valid_from=datetime(2025, 11, 14),
       valid_until=datetime(2026, 1, 20),  # <--- ADD THIS
       distances_mm={1: 8855.0, 2: 9455.0, 3: 12544.0, 4: 12545.0},
       notes="Configuration from Nov 14 to Jan 20, 2026"
   ),
   
   # Add new config:
   ProbeConfiguration(
       name="jan2026_recalibration",
       valid_from=datetime(2026, 1, 20),
       valid_until=None,  # <--- None = current config
       distances_mm={
           1: 8900.0,  # <--- YOUR NEW MEASURED VALUES
           2: 9500.0,
           3: 12600.0,
           4: 12601.0,
       },
       notes="Moved probes after tank maintenance Jan 20, 2026"
   ),

2. That's it! No other code changes needed.

The system will automatically:
- Use correct positions for old files (before Jan 20)
- Use new positions for new files (after Jan 20)
- Work if you reprocess old data
"""


# =============================================================================
# BENEFITS OF THIS APPROACH
# =============================================================================

"""
COMPARED TO YOUR CURRENT CODE:

BEFORE (problems):
❌ Hardcoded dates scattered in logic (distance_cutoff, mooring_cutoff)
❌ Need to modify if-statements every time positions change
❌ Easy to forget to update all the places
❌ Hard to see history of changes
❌ String date comparison bugs ("20251106")
❌ Duplicate logic for date extraction

AFTER (benefits):
✅ All configurations in one place
✅ Adding new configs is trivial
✅ Clear history of all changes
✅ Proper datetime comparison
✅ Self-documenting code
✅ Easy to test ("what positions on this date?")
✅ No risk of forgetting to update something

REAL WORLD EXAMPLE:

You have a file: "20251115-fullpanel-amp02-freq13-per40.csv"
Date: November 15, 2025

Old code:
  file_date < datetime(2025, 11, 14)?  → False
  → Returns None for all probe positions (BUG!)

New code:
  get_probe_positions(datetime(2025, 11, 15))
  → Finds "nov14_adjustment" config (valid from Nov 14)
  → Returns {1: 8855.0, 2: 9455.0, ...}
  → Correct!
"""


# =============================================================================
# MINIMAL CHANGES TO YOUR EXISTING FILE
# =============================================================================

"""
If you want to make MINIMAL changes to your current code:

1. Add these 3 things at the top of data_loader.py:
   - ProbeConfiguration dataclass
   - PROBE_CONFIGS list
   - get_probe_positions() function
   - _extract_file_date() function

2. Replace this block (lines ~180-186):
   
   # OLD:
   distance_cutoff = datetime(2025, 11, 14)
   metadata["Probe 1 mm from paddle"] = 8855 if file_date < distance_cutoff else None
   # ... etc
   
   # NEW:
   file_date = _extract_file_date(filename, path)
   if file_date:
       probe_positions = get_probe_positions(file_date)
       for probe_num, distance in probe_positions.items():
           metadata[f"Probe {probe_num} mm from paddle"] = distance

3. Do the same for mooring (lines ~165-175)

That's it! Only ~30 lines of new code, replace ~20 lines of old code.
Total change: +10 lines, but MUCH more maintainable.
"""


# =============================================================================
# TESTING YOUR CHANGES
# =============================================================================

"""
After making changes, test like this:

# At bottom of data_loader.py
if __name__ == "__main__":
    # Test probe positions
    test_dates = [
        datetime(2025, 10, 1),   # Before Nov 14
        datetime(2025, 11, 14),  # Exactly Nov 14
        datetime(2025, 12, 1),   # After Nov 14
    ]
    
    for date in test_dates:
        positions = get_probe_positions(date)
        print(f"{date.date()}: Probe 1 = {positions[1]} mm")

Expected output:
  2025-10-01: Probe 1 = 8855.0 mm
  2025-11-14: Probe 1 = 8855.0 mm  (or new value if you updated it)
  2025-12-01: Probe 1 = 8855.0 mm  (or new value if you updated it)
"""
