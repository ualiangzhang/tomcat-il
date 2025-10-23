"""
Excel map parser for ASIST Saturn maps.

This module parses two maps contained in `data/map_excel/asist_map.xlsx` and
produces two numpy arrays for MiniGrid-compatible maps. The expected sheet
names are `SaturnA_2.3` and `SaturnB_2.3`.

Changes from the legacy parser:
- Switch to `openpyxl` to read modern .xlsx files (xlrd dropped xlsx support).
- Support for new symbols: A/B/C victims, X collapse plate/threat collapse,
  D falling rubble, P victim detection plate, R/RR/RRR rubble layers,
  T freezing threat, F objects.
- Grey and brown filled cells are considered walls.
- No doors are produced anymore.
- Map extents are larger, spanning coordinates from (-2226,-13) in the
  top-left to (-2087, 64) in the bottom-right. This results in a fixed grid
  of height 78 and width 140.

Output files are saved under `gym_minigrid/envs/resources/` with names:
- `SaturnA_2_3.npy`
- `SaturnB_2_3.npy`
- `raw_map_state_saturnA.npy`, `raw_map_state_saturnB.npy` (walkable==1, walls==4)

All functions are annotated with type hints and in-line comments describe
non-obvious implementation details.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from openpyxl import load_workbook
from openpyxl.cell.cell import Cell
try:
    # Older/newer openpyxl versions expose COLOR_INDEX here; fallback to empty
    from openpyxl.styles.colors import COLOR_INDEX  # type: ignore
except Exception:  # pragma: no cover - best-effort compatibility
    COLOR_INDEX = {}  # type: ignore


# Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "map_excel"
RESOURCES_DIR = (Path(__file__).parent / "./envs/resources").resolve()

# Workbook and sheet configuration
WORKBOOK_PATH = DATA_DIR / "asist_map.xlsx"
SHEET_NAMES = ("SaturnA_2.3", "SaturnB_2.3")

# Coordinate system (inclusive extents)
TOP_LEFT: Tuple[int, int] = (-2226, -13)
BOTTOM_RIGHT: Tuple[int, int] = (-2087, 64)

# Derived grid size (height x width)
GRID_HEIGHT: int = BOTTOM_RIGHT[1] - TOP_LEFT[1] + 1  # 78
GRID_WIDTH: int = BOTTOM_RIGHT[0] - TOP_LEFT[0] + 1   # 140


# MiniGrid integer IDs used in this repository (see gym_minigrid/index_mapping.py)
EMPTY = 1
WALL = 4           # Default wall
WALL_HEAVY = 30    # Alternative wall id (rendered as wall)
LAVA = 9           # Used for threats/hazards
BOX = 255          # Generic interactable/plate/object
GOAL_A = 81        # Victim A
GOAL_B = 82        # Victim B
GOAL_C = 83        # Victim C


# Colors used in the Excel for wall identification via fill
# These are common RGB hex strings (uppercase) for grey/brown families.
GREY_HEXES: Iterable[str] = {
    "FF808080", "FF7F7F7F", "FFC0C0C0", "FFBFBFBF", "FFB0B0B0"
}
BROWN_HEXES: Iterable[str] = {
    "FFCBA986", "FFA8947D", "FF8B4513", "FFA0522D", "FFCD853F"
}


def _color_to_argb(cell: Cell) -> Optional[str]:
    """Return the ARGB hex string for the cell fill color if present.

    openpyxl may encode colors as RGB, indexed palette, or theme-based.
    This helper normalizes to an ARGB string when possible.
    """
    fill = cell.fill
    if fill is None or fill.patternType is None:
        return None

    color = fill.start_color
    if color is None:
        return None

    # RGB specified directly
    if getattr(color, "type", None) == "rgb" and color.rgb:
        return color.rgb.upper()

    # Indexed palette
    if getattr(color, "type", None) == "indexed" and color.indexed is not None:
        try:
            argb = COLOR_INDEX[color.indexed]
            return (argb or "").upper() or None
        except Exception:
            return None

    # Theme-based colors are hard to resolve without workbook theme; ignore
    return None


def _is_wall_fill(cell: Cell) -> bool:
    """Heuristic: consider filled grey/brown cells as walls.

    Returns True if the cell's fill color matches one of the configured
    grey/brown ARGB values.
    """
    argb = _color_to_argb(cell)
    if not argb:
        return False
    return argb in GREY_HEXES or argb in BROWN_HEXES


def _normalize_token(val: Optional[str]) -> str:
    """Normalize cell textual content for symbol matching.

    - Strips whitespace
    - Upper-cases ASCII letters
    - Returns empty string for None
    """
    if val is None:
        return ""
    return str(val).strip().upper()


def symbol_to_minigrid(token: str) -> int:
    """Map a normalized token to a MiniGrid integer id.

    The mapping follows project conventions while reusing existing ids
    to remain compatible with `gym_minigrid.numpymap` rendering:
    - A, B, C: different victim goals (81/82/83)
    - X: collapse plate / threat collapse → hazard (lava id 9)
    - D: falling rubble → hazard (lava id 9)
    - T: freezing threat → hazard (lava id 9)
    - P: victim detection plate → interactable plate (box id 255)
    - F: object → generic object (box id 255)
    - R, RR, RRR: rubble layers → walls (4 or 30 for heavier rubble)
    - empty/other: walkable empty (1)
    """
    if token == "A":
        return GOAL_A
    if token == "B":
        return GOAL_B
    if token == "C":
        return GOAL_C
    if token in {"X"}:
        return LAVA
    if token in {"D", "T"}:
        return LAVA
    if token in {"P", "F"}:
        return BOX
    if token == "RRR":
        return WALL_HEAVY
    if token == "RR":
        return WALL_HEAVY
    if token == "R":
        return WALL
    return EMPTY


def find_content_origin(sheet) -> Tuple[int, int]:
    """Best-effort detection of the first map row/column.

    Many mapping sheets include header rows/columns. We scan top-left area
    to find the earliest row and column that contain either a known symbol
    or a filled wall cell.

    Returns (row_start, col_start) as 1-based indices for openpyxl.
    """
    max_scan_rows = min(50, sheet.max_row)
    max_scan_cols = min(50, sheet.max_column)

    row_start = None
    col_start = None

    for r in range(1, max_scan_rows + 1):
        for c in range(1, max_scan_cols + 1):
            cell = sheet.cell(row=r, column=c)
            token = _normalize_token(cell.value)
            if token in {"A", "B", "C", "X", "D", "P", "R", "RR", "RRR", "T", "F"} or _is_wall_fill(cell):
                row_start = r if row_start is None else min(row_start, r)
                col_start = c if col_start is None else min(col_start, c)
    if row_start is None or col_start is None:
        # Fallback to 1,1 if detection fails
        row_start = 1
        col_start = 1
    return row_start, col_start


def parse_saturn_sheet(sheet_name: str, save_basename: str) -> None:
    """Parse one Saturn sheet into a MiniGrid numpy array and save it.

    Parameters
    ----------
    sheet_name: str
        Name of the sheet in the workbook to parse.
    save_basename: str
        Basename for the output files (without extension). Two files are
        produced: `<basename>.npy` and `raw_map_state_<suffix>.npy`.
    """
    wb = load_workbook(WORKBOOK_PATH, data_only=True)
    if sheet_name not in wb.sheetnames:
        raise ValueError(f"Sheet '{sheet_name}' not found in {WORKBOOK_PATH}")
    sheet = wb[sheet_name]

    # Initialize grid (row-major: y, x)
    grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int32) + EMPTY

    row_start, col_start = find_content_origin(sheet)

    for r in range(GRID_HEIGHT):
        for c in range(GRID_WIDTH):
            cell = sheet.cell(row=row_start + r, column=col_start + c)
            token = _normalize_token(cell.value)

            # First, prefer explicit symbol; if not present, use fill color
            val = symbol_to_minigrid(token)
            if val == EMPTY and _is_wall_fill(cell):
                val = WALL
            grid[r, c] = val

    # Add a solid wall border for safety (consistent with legacy behavior)
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL

    # Build raw_map_state: walkable=1, walls=4; goals/boxes remain walkable
    raw_map_state = (grid.copy()).astype(np.int32)
    raw_map_state[raw_map_state == BOX] = EMPTY
    raw_map_state[raw_map_state == GOAL_A] = EMPTY
    raw_map_state[raw_map_state == GOAL_B] = EMPTY
    raw_map_state[raw_map_state == GOAL_C] = EMPTY
    raw_map_state[raw_map_state == LAVA] = WALL  # treat hazards as blocked in state
    raw_map_state[raw_map_state == WALL_HEAVY] = WALL

    # Save outputs
    RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
    out_map = RESOURCES_DIR / f"{save_basename}.npy"
    np.save(out_map, grid)

    out_state = RESOURCES_DIR / f"raw_map_state_{save_basename.split('_')[0].lower()}.npy"
    np.save(out_state, raw_map_state)

    print(f"Saved: {out_map}")
    print(f"Saved: {out_state}")


def main() -> None:
    """Entry point: parse both Saturn maps and save outputs.

    This function is idempotent and will overwrite existing files.
    """
    if not WORKBOOK_PATH.exists():
        raise FileNotFoundError(f"Workbook not found: {WORKBOOK_PATH}")

    sheet_to_out: Dict[str, str] = {
        "SaturnA_2.3": "SaturnA_2_3",
        "SaturnB_2.3": "SaturnB_2_3",
    }

    for sheet_name in SHEET_NAMES:
        parse_saturn_sheet(sheet_name, sheet_to_out[sheet_name])


if __name__ == "__main__":
    main()
