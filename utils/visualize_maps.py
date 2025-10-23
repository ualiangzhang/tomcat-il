"""
Visualize generated MiniGrid maps as PNGs for quick inspection.

This script loads the Saturn maps saved under
`gym_minigrid/envs/resources/` and writes colorized images to
`gym_minigrid/envs/resources/vis/`.

Run:
    python -m utils.visualize_maps
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


RESOURCES_DIR = (Path(__file__).parent.parent / "gym_minigrid" / "envs" / "resources").resolve()
OUT_DIR = RESOURCES_DIR / "vis"


def show_and_save(name: str) -> None:
    arr = np.load(RESOURCES_DIR / f"{name}.npy")
    plt.figure(figsize=(10, 6))
    plt.imshow(arr, interpolation="nearest")
    plt.title(name)
    plt.colorbar(shrink=0.6)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved visualization: {out_path}")


def main() -> None:
    for base in ["SaturnA_2_3", "SaturnB_2_3", "raw_map_state_saturna", "raw_map_state_saturnb"]:
        show_and_save(base)


if __name__ == "__main__":
    main()


