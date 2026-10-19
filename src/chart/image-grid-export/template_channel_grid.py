"""
Template: archetype image grids for ONE stain/channel.

Generalised from the per-channel scripts in ./channel_scripts/ (all share this
logic; they differ only in the values filled in under "FILL IN" below).

For each archetype:
  1. Load the archetype's cell table (one row per cell: Well, cell label,
     cell bounding box, and an `Archetype<N>` weight column).
  2. Rank cells by their archetype weight and keep the top N_TOP_CELLS.
  3. Split them into rows of CELLS_PER_GRID and save one PNG per row.
  For each cell panel:
     - crop the registered phenotype image around the cell bbox (+ padding)
     - display the stain channel in inverted grayscale with that well's
       vmin/vmax (per-well display normalisation)
     - overlay the cell mask contour (red) and nucleus mask contour (blue)
     - label axes with crop size in pixels; title with archetype, weight,
       well and cell ID.

To set up a new channel: copy this file, fill in every <...> placeholder,
then run it with `python <your_copy>.py`.
"""
import os

import matplotlib
matplotlib.use("Agg")  # save to file only, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scallops.io import read_image

# =============================================================================
# FILL IN: stain / output
# =============================================================================
STAIN = "<STAIN_NAME>"            # used in the input path and the output file name
SAVE_DIR = "<OUTPUT_DIR>"         # one folder per stain

# Per-cell archetype table; must contain `{archetype_n}`.
# Expected columns: Well, LABEL_COL, cell_bbox-0..3, Archetype<N>
ARCHETYPE_TABLE_PATH = "<PATH_TO_ARCHETYPE_TABLES>/{stain}/Archetype{archetype_n}_cellcoordinates.parquet"
LABEL_COL = "Label"               # some tables use "label"

ARCHETYPE_RANGE = range(1, 20)    # missing archetypes are skipped automatically

# =============================================================================
# FILL IN: display settings per well
# (channel index in the image, vmin, vmax, display name)
# Pick vmin/vmax per well so crops are comparable across wells.
# =============================================================================
CHANNEL_INFO = {
    "<WELL_1>": (0, 0, 0, STAIN),   # (channel_idx, vmin, vmax, name)
    "<WELL_2>": (0, 0, 0, STAIN),
    # ...
}

# =============================================================================
# FILL IN: image / mask locations per well
# Each entry is a format string with `{well}`; add a per-well override in the
# *_OVERRIDES dicts when a well lives in a different zarr.
# =============================================================================
IMAGE_PATH = "<REGISTERED_PHENO_ZARR>/images/{well}"
CELL_MASK_PATH = "<CELL_SEG_ZARR>/labels/{well}-cell"
NUC_MASK_PATH = "<NUC_SEG_ZARR>/labels/{well}-nuclei"

IMAGE_PATH_OVERRIDES = {}       # e.g. {"<WELL>": "<OTHER_ZARR>/images/{well}"}
CELL_MASK_PATH_OVERRIDES = {}
NUC_MASK_PATH_OVERRIDES = {}

# =============================================================================
# Layout (values used in the original scripts)
# =============================================================================
N_TOP_CELLS = 15
CELLS_PER_GRID = 5
BBOX_PAD = 8                    # pixels added on each side of the cell bbox
CMAP = "gray_r"
CELL_CONTOUR_COLOR = "red"
NUC_CONTOUR_COLOR = "blue"
CONTOUR_ALPHA = 0.5
MASK_LINEWIDTH = 3
PIXEL_LABEL_FONTSIZE = 18
TITLE_FONTSIZE = 8
DPI = 300

np.random.seed(42)


def resolve(path_template, overrides, well):
    return overrides.get(well, path_template).format(well=well)


def plot_cell(ax, row, archetype_n, arch_col):
    """Draw one cell panel. Returns False if the cell had to be skipped."""
    well = row["Well"]
    label = row[LABEL_COL]

    try:
        y0 = int(row["cell_bbox-0"]) - BBOX_PAD
        y1 = int(row["cell_bbox-2"]) + BBOX_PAD
        x0 = int(row["cell_bbox-1"]) - BBOX_PAD
        x1 = int(row["cell_bbox-3"]) + BBOX_PAD
        if y1 <= y0 or x1 <= x0:
            print(f"Invalid bbox for {well}-{label}, skipping.")
            return False
    except Exception as e:
        print(f"Error parsing bbox for {well}-{label}: {e}")
        return False

    if well not in CHANNEL_INFO:
        print(f"Missing channel config for {well}, skipping.")
        return False
    c, vmin, vmax, name = CHANNEL_INFO[well]

    try:
        image = read_image(resolve(IMAGE_PATH, IMAGE_PATH_OVERRIDES, well), dask=True).squeeze()
        cell_labels = read_image(resolve(CELL_MASK_PATH, CELL_MASK_PATH_OVERRIDES, well), dask=True)
        nuc_labels = read_image(resolve(NUC_MASK_PATH, NUC_MASK_PATH_OVERRIDES, well), dask=True)
    except Exception as e:
        print(f"Error loading image/masks for {well}: {e}")
        return False

    # Cell and nucleus share the same label ID
    crop = dict(y=slice(y0, y1), x=slice(x0, x1))
    image_slice = image.isel(c=c, **crop)
    cell_mask = (cell_labels == label).isel(**crop)
    nuc_mask = (nuc_labels == label).isel(**crop)

    ax.imshow(image_slice, cmap=CMAP, vmin=vmin, vmax=vmax)
    ax.contour(cell_mask, colors=CELL_CONTOUR_COLOR, linewidths=MASK_LINEWIDTH, levels=[0.5], alpha=CONTOUR_ALPHA)
    ax.contour(nuc_mask, colors=NUC_CONTOUR_COLOR, linewidths=MASK_LINEWIDTH, levels=[0.5], alpha=CONTOUR_ALPHA)

    ax.axis("on")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"{x1 - x0} pt", fontsize=PIXEL_LABEL_FONTSIZE)
    ax.set_ylabel(f"{y1 - y0} pt", fontsize=PIXEL_LABEL_FONTSIZE)
    ax.set_title(
        f"Archetype {archetype_n} | weight: {row[arch_col]:.3f} | Well: {well} | Cell ID: {label}",
        fontsize=TITLE_FONTSIZE,
    )
    return True


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    for archetype_n in ARCHETYPE_RANGE:
        print(f"\nProcessing Archetype {archetype_n}...")
        table_path = ARCHETYPE_TABLE_PATH.format(stain=STAIN, archetype_n=archetype_n)
        try:
            df = pd.read_parquet(table_path).reset_index()
            print(f"Loaded {len(df)} cells for Archetype {archetype_n}")
        except Exception as e:
            print(f"Error loading data for Archetype {archetype_n}: {e}")
            continue

        # Top cells by archetype weight
        arch_col = f"Archetype{archetype_n}"
        n = min(N_TOP_CELLS, len(df))
        top = df.sort_values(by=arch_col, ascending=False).head(n)
        print(f"Selected top {n} cells ranked by {arch_col}")

        groups = [top.iloc[i:i + CELLS_PER_GRID] for i in range(0, n, CELLS_PER_GRID)]
        for group_idx, group in enumerate(groups, 1):
            fig, axs = plt.subplots(1, len(group), figsize=(5 * len(group), 5))
            axs = np.atleast_1d(axs)
            for ax, (_, row) in zip(axs, group.iterrows()):
                plot_cell(ax, row, archetype_n, arch_col)

            plt.tight_layout()
            out_name = f"archetype{archetype_n}_{STAIN}_grid_{group_idx}.png"
            plt.savefig(os.path.join(SAVE_DIR, out_name), dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved grid image: {out_name}")

    print(f"\nAll grid images saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
