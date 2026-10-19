# Archetype image grids

Code for making image grids of the cells that best represent each
morphological archetype, one stain at a time. Each grid image shows single-cell
crops of the top-ranked cells for an archetype. The stain is shown in grayscale
with the cell and nucleus outlines on top.

## What's in this folder

```
├── README.md
├── template_display_range_per_well.ipynb   # step 1: choose display ranges
├── template_channel_grid.py                # step 2: make the grids
└── channel_scripts/                        # the scripts used for our figures (examples)
```

The two templates contain no real paths or values. Every setting you need to
fill in is marked `<...>`.

## Inputs you need

1. **Registered phenotype images**, one per well, readable with
   `scallops.io.read_image`. They should have dimensions `(c, y, x)`.
2. **Cell and nucleus label images**, one each per well. A cell and its
   nucleus must share the same label ID.
3. **Archetype cell tables**, one parquet file per stain and archetype, with
   these columns:
   - `Well`
   - the cell label (`Label` or `label`; set `LABEL_COL`)
   - the cell bounding box: `cell_bbox-0` (y min), `cell_bbox-1` (x min),
     `cell_bbox-2` (y max), `cell_bbox-3` (x max)
   - `Archetype<N>`: the cell's weight for archetype N.

## Step 1: choose display ranges

`template_display_range_per_well.ipynb` sets `vmin` and `vmax` for each well.

### Why per well

Staining intensity varies from well to well. With one absolute `vmin`/`vmax`
for every well, the same biology would look brighter in some wells than
others. Instead:

1. For each channel, choose **one pair of pixel percentiles** that gives a
   good visualisation.
   - The low percentile sets `vmin`.
   - The high percentile sets `vmax`.
   - Bright stains usually need a lower high percentile, around 99. Weak or
     sparse stains need a higher one, 99.8–99.9.
2. **Keep those percentiles the same for every well**, and compute
   `vmin`/`vmax` separately in each well.

This normalises the display per well, so crops can be compared between wells.

### How the notebook computes them

- **Sampling:** it pools pixels from five large patches (centre + four
  quadrants) inside the central part of each well. This avoids edges and empty
  margins. You can change `PATCH_SIZE` and `MARGIN_FRAC`.
- **Output:** it prints one line per well, ready to paste:
  `channel_info_<WELL> = [(channel_idx, vmin, vmax, name), ...]`.
- **Visual check:** the last cell shows the same channel side by side for every
  well, each with its own range. Adjust the percentiles until the wells look
  comparable. Then copy the values into the grid script.

Use the **same image source** in the notebook and the grid script, so the
ranges are computed on the images that are actually displayed.

## Step 2: make the grids

`template_channel_grid.py` handles one stain per copy. To set it up:

1. Copy it, e.g. to `MYSTAIN.py`.
2. Fill in:

| Setting | What it is |
|---|---|
| `STAIN` | Stain name, used in the input table path and output file names |
| `SAVE_DIR` | Output folder |
| `ARCHETYPE_TABLE_PATH` | Path pattern for the archetype tables, with `{stain}` and `{archetype_n}` |
| `LABEL_COL` | Name of the cell-label column |
| `ARCHETYPE_RANGE` | Archetype numbers to try; missing ones are skipped |
| `CHANNEL_INFO` | For each well: `(channel_idx, vmin, vmax, name)` from Step 1 |
| `IMAGE_PATH`, `CELL_MASK_PATH`, `NUC_MASK_PATH` | Path patterns with `{well}` |
| `*_OVERRIDES` | Per-well exceptions, for wells stored somewhere else |

3. Run it: `python MYSTAIN.py`

### What the script does for each archetype

1. Loads the archetype's cell table.
2. Ranks the cells by `Archetype<N>` weight and keeps the top `N_TOP_CELLS`
   (15).
3. Splits them into rows of `CELLS_PER_GRID` (5), one PNG per row:
   `archetype<N>_<STAIN>_grid_<k>.png`.
4. Builds each cell panel:
   - crops the image around the cell's bounding box, with `BBOX_PAD` (8 px)
     of padding;
   - shows the stain channel in inverted grayscale (`gray_r`), using that
     well's `vmin`/`vmax`;
   - draws the cell outline in red and the nucleus outline in blue;
   - labels the axes with the crop size in pixels;
   - sets the title to the archetype, weight, well and cell ID.

The layout constants (cell count, padding, colours, line width, dpi) are at the
top of the script. They're set to the values used for our figures.

## Example scripts

`channel_scripts/` has the per-channel scripts used for our figures. They
follow the same logic as the template, written out with the actual values.
They're for reference only, because their paths point at our storage.
