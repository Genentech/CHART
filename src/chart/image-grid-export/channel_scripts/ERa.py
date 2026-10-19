import os
import pandas as pd
import matplotlib.pyplot as plt
from scallops.io import read_image
from scallops.visualize import imcomposite
import numpy as np

save_dir = "/data/Paula/ePC057/mar2026/output2_lines/ERa/"
os.makedirs(save_dir, exist_ok=True)
stain = "ERa"
PIXEL_LABEL_FONTSIZE = 18
MASK_LINEWIDTH = 3

# -------------------------------
# Channel configuration (keeping DAPI channels only)
# -------------------------------
channel_info_A1 = [(18, 112, 1000, 'ERa')]
channel_info_A2 = [(18, 114, 1000, 'ERa')]
channel_info_A3 = [(18, 113, 1000, 'ERa')]
channel_info_B1 = [(18, 112, 1000, 'ERa')]
channel_info_B2 = [(18, 113, 1000, 'ERa')]
channel_info_B3 = [(18, 114, 1000, 'ERa')]
                                         
channel_info_map = {
    "A1": channel_info_A1,
    "A2": channel_info_A2,
    "A3": channel_info_A3,
    "B1": channel_info_B1,
    "B2": channel_info_B2,
    "B3": channel_info_B3
}

# Set random seed for reproducibility
np.random.seed(42)

# -------------------------------
# Loop over archetypes
# -------------------------------
for archetype_n in range(1, 20):
    print(f"\nProcessing Archetype {archetype_n}...")

    bbox_path = f's3://bigdipir-ctg-s3/internal/singa166-lab/Carolina/cellmapp/archetypes_test/robust_archetypes/{stain}/Archetype{archetype_n}_cellcoordinates.parquet'
    
    try:
        df_bbox = pd.read_parquet(bbox_path).reset_index()
        print(f"Loaded {len(df_bbox)} cells for Archetype {archetype_n}")
    except Exception as e:
        print(f"Error loading data for Archetype {archetype_n}: {e}")
        continue

    # -------------------------------
    # Rank cells by archetype weight
    # -------------------------------
    arch_col = f'Archetype{archetype_n}'
    df_sorted = df_bbox.sort_values(by=arch_col, ascending=False)

    # Select top 15 cells
    n_samples = min(15, len(df_sorted))
    df_sampled = df_sorted.head(n_samples)
    print(f"Selected top {n_samples} cells ranked by {arch_col}")

    # Split into groups of 5
    grouped = [df_sampled.iloc[i:i+5] for i in range(0, n_samples, 5)]

    for group_idx, group in enumerate(grouped, 1):
        fig, axs = plt.subplots(1, len(group), figsize=(5 * len(group), 5))
        if len(group) == 1:
            axs = [axs]  # Handle single image case

        for ax, (_, row) in zip(axs, group.iterrows()):
            well = row['Well']
            label = row['label']
            arch_weight = row[arch_col]

            try:
                y0 = int(row['cell_bbox-0']) - 8
                y1 = int(row['cell_bbox-2']) + 8
                x0 = int(row['cell_bbox-1']) - 8
                x1 = int(row['cell_bbox-3']) + 8
                if y1 <= y0 or x1 <= x0:
                    print(f"Invalid bbox for {well}-{label}, skipping.")
                    continue
            except Exception as e:
                print(f"Error parsing bbox for {well}-{label}: {e}")
                continue

            channel_info = channel_info_map.get(well)
            if channel_info is None:
                print(f"Missing channel config for {well}, skipping.")
                continue

            # Load image
            try:
                if well == "A1":
                    image = read_image(
                        f's3://bigdipir-ctg-s3/internal/singa166-lab/Paula/ePC057/analysis/pheno-registration-Carolina/non-linear/p-registered-to-p4-unroll.zarr/images/{well}',
                        dask=True).squeeze()
                elif well == "A2":
                    image = read_image(
                        f's3://bigdipir-ctg-s3/internal/singa166-lab/Paula/ePC057/analysis/new2025_wellA2/pheno-regs.zarr/images/{well}',
                        dask=True).squeeze()
                else:
                    image = read_image(
                        f's3://bigdipir-ctg-s3/internal/singa166-lab/Paula/ePC057/analysis/pheno-registration-Carolina/p-registered-to-p4-unroll.zarr/images/{well}',
                        dask=True).squeeze()
            except Exception as e:
                print(f"Error loading image for {well}: {e}")
                continue

            # Load cell mask
            try:
                if well == "A2":
                    label_image = read_image(
                        f's3://bigdipir-ctg-s3/internal/singa166-lab/Paula/ePC057/analysis/new2025_wellA2/cell-seg.zarr/labels/{well}-cell',
                        dask=True)
                else:
                    label_image = read_image(
                        f's3://bigdipir-ctg-s3/internal/singa166-lab/Paula/ePC057/analysis/pheno-cell-seg-propagation-p4/Plate1A.zarr/labels/{well}-cell',
                        dask=True)
            except Exception as e:
                print(f"Error loading label for {well}: {e}")
                continue

            label_mask = (label_image == label)

            # Load nuclei mask
            try:
                if well == "A2":
                    nuclei_image = read_image(
                        f's3://bigdipir-ctg-s3/internal/singa166-lab/Paula/ePC057/analysis/new2025_wellA2/nuc-seg.zarr/labels/{well}-nuclei',
                        dask=True)
                else:
                    nuclei_image = read_image(
                        f's3://bigdipir-ctg-s3/internal/singa166-lab/Paula/ePC057/analysis/pheno-segment-stardist-p4/Plate1A.zarr/labels/{well}-nuclei',
                        dask=True)
            except Exception as e:
                print(f"Error loading label for {well}: {e}")
                continue

            nuc_mask = (nuclei_image == label)

            # Plotting
            try:
                c, vmin, vmax, name = channel_info[0]
                image_slice = image.isel(c=c, y=slice(y0, y1), x=slice(x0, x1))
                mask_slice = label_mask.isel(y=slice(y0, y1), x=slice(x0, x1))
                nuc_slide = nuc_mask.isel(y=slice(y0, y1), x=slice(x0, x1))

                # 1. Base image
                ax.imshow(image_slice, cmap='gray_r', vmin=vmin, vmax=vmax)

                # 2. Contours
                ax.contour(mask_slice, colors='red', linewidths=MASK_LINEWIDTH, levels=[0.5], alpha=0.5)
                ax.contour(nuc_slide, colors='blue', linewidths=MASK_LINEWIDTH, levels=[0.5], alpha=0.5)

                # 3. Annotations
                x_len = x1 - x0
                y_len = y1 - y0
                ax.axis('on')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(f'{x_len} pt', fontsize=PIXEL_LABEL_FONTSIZE)
                ax.set_ylabel(f'{y_len} pt', fontsize=PIXEL_LABEL_FONTSIZE)
                ax.set_title(
                    f"Archetype {archetype_n} | weight: {arch_weight:.3f} | Well: {well} | Cell ID: {label}",
                    fontsize=8
                )

                plt.tight_layout()
                plt.show()

            except Exception as e:
                print(f"Plotting failed for {well}-{label}: {e}")
                continue

        plt.tight_layout()
        grid_filename = f"archetype{archetype_n}_{stain}_grid_{group_idx}.png"
        plt.savefig(os.path.join(save_dir, grid_filename), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved grid image: {grid_filename}")

print(f"\nAll grid images saved to: {save_dir}")