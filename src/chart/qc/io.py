"""
Input/output functions for quality control analysis.

This module contains functions for reading registered images and object
tables from disk and for writing filter files, so that the analysis
modules stay free of I/O concerns.
"""

import os
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Import scallops for image processing
try:
    import scallops
    from scallops import io
    SCALLOPS_AVAILABLE = True
except ImportError:
    # This happens at import time, before an entry point has configured any
    # handlers, so it reaches the console but not a log file.
    logger.warning("scallops not available. Image processing features will be disabled.")
    SCALLOPS_AVAILABLE = False


def load_pheno_dapi_images(pheno_dir, well, dapi_channel_names):
    """Load registered pheno images, restricted to the DAPI channels."""
    if_reg = scallops.io.read_experiment(pheno_dir, dask=True, group_by=None)
    if_reg = if_reg.images[well]
    if_reg = if_reg.sel(c=dapi_channel_names).compute()
    return if_reg


def load_pheno_to_sbs_dapi_image(pheno_to_sbs_dir, well, pheno_dapi_channel):
    """Load the pheno DAPI image registered into SBS space."""
    if pheno_dapi_channel is None:
        raise ValueError("pheno_dapi_channel is required; set 'pheno_sbs_dapi_channel' "
                         "or 'dapi_channel_names' in the configuration")

    pheno_reg = scallops.io.read_experiment(pheno_to_sbs_dir, dask=True, group_by=None)
    pheno_reg = pheno_reg.images[well]
    if isinstance(pheno_dapi_channel, int):
        logger.info(f"Selecting pheno DAPI channel by index: {pheno_dapi_channel}")
        return pheno_reg.isel(c=pheno_dapi_channel).compute()
    logger.info(f"Selecting pheno DAPI channel by name: {pheno_dapi_channel}")
    return pheno_reg.sel(c=pheno_dapi_channel).compute()


def load_sbs_dapi_image(sbs_dir, well):
    """Load the SBS DAPI image (channel 0)."""
    sbs = scallops.io.read_experiment(sbs_dir, dask=True, group_by=None)
    sbs = sbs.images[well]
    sbs = sbs.isel(c=0).compute()
    return sbs


def load_objects(objects_dir, well, premerged=False, column_mapping=None):
    """Load object data for a specific well.
    
    Args:
        objects_dir: Directory containing object/merged files
        well: Well identifier
        premerged: If True, read {well}.parquet (upstream pipeline already merged
                   objects + features + reads). If False, read {well}-objects.parquet.
        column_mapping: Optional dict mapping source column names to cellmapp
                        standard names (e.g. ``{"barcode_0": "sgRNA"}``).
                        Applied after loading so downstream code can use a
                        consistent vocabulary.  When ``None`` no renaming is done.
    """
    suffix = "" if premerged else "-objects"
    path = os.path.join(objects_dir, f"{well}{suffix}.parquet")
    logger.info(f"Reading object matrix from {path}")
    objects = pd.read_parquet(path)

    if column_mapping:
        present = {k: v for k, v in column_mapping.items() if k in objects.columns}
        if present:
            objects = objects.rename(columns=present)
            logger.info(f"Renamed {len(present)} columns via column_mapping")

    if 'label' in objects.columns:
        objects = objects.set_index('label', drop=False)
    else:
        objects.index.name = 'label'
        objects['label'] = objects.index
    logger.info(f"Loaded {len(objects)} objects")
    return objects


def load_sbs_objects(sbs_feature_dir, well, objects):
    """Load object data from registered SBS space, aligned to *objects*."""
    sbs_objects = pd.read_parquet(os.path.join(sbs_feature_dir, "nuclei", f"{well}-objects.parquet"))
    sbs_objects = sbs_objects.reindex(objects.index)
    sbs_objects = sbs_objects.dropna(how='all')
    return sbs_objects


def load_precomputed_filters(precomputed_config, well, output_dir):
    """Load precomputed filter files from an upstream pipeline and save locally."""
    if not precomputed_config:
        return

    os.makedirs(output_dir, exist_ok=True)
    for filter_name, cfg in precomputed_config.items():
        file_pattern = cfg.get('file_pattern', '{well}-objects.parquet')
        filter_path = os.path.join(cfg['path'], file_pattern.format(well=well))
        threshold_col = cfg.get('threshold_column', None)
        threshold_val = cfg.get('threshold', None)
        logger.info(f"Loading precomputed filter '{filter_name}' from {filter_path}")
        try:
            df = pd.read_parquet(filter_path)
            total = len(df)

            if threshold_col and threshold_val is not None:
                df = df.dropna(subset=[threshold_col])
                df = df[df[threshold_col] >= threshold_val]
                logger.info(f"  Thresholded {threshold_col} >= {threshold_val}: "
                            f"{len(df)}/{total} pass ({len(df)/total*100:.1f}%)")

            labels = pd.DataFrame({'label': df.index})
            out_path = os.path.join(output_dir, f"{well}_{filter_name}.parquet")
            labels.to_parquet(out_path)
            logger.info(f"  Saved {len(labels)} passing labels → {out_path}")
        except Exception as e:
            logger.warning(f"  could not load precomputed filter '{filter_name}': {e}")


def save_filters(phenocor_labels, phenosbscor_labels, well, output_dir):
    """Save filter labels to parquet files."""
    os.makedirs(output_dir, exist_ok=True)
    
    if phenocor_labels is not None:
        phenocor_df = pd.DataFrame(phenocor_labels)
        phenocor_path = os.path.join(output_dir, f"{well}_phenocorfilt.parquet")
        phenocor_df.to_parquet(phenocor_path)
        logger.info(f"Saved pheno correlation filter: {phenocor_path}")
    
    if phenosbscor_labels is not None:
        phenosbscor_df = pd.DataFrame(phenosbscor_labels)
        phenosbscor_path = os.path.join(output_dir, f"{well}_phenosbscorfilt.parquet")
        phenosbscor_df.to_parquet(phenosbscor_path)
        logger.info(f"Saved pheno-SBS correlation filter: {phenosbscor_path}")


def save_segmentation_filters(high_ratio_errors: pd.Series,
                            well: str,
                            output_dir: str) -> None:
    """
    Save segmentation error filters to parquet files.
    
    Args:
        high_ratio_errors: Boolean Series for high ratio errors
        well: Well identifier
        output_dir: Directory to save filter files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save high ratio errors filter
    if high_ratio_errors.any():
        # Create DataFrame with 'label' column containing the problematic labels
        problematic_labels = high_ratio_errors[high_ratio_errors].index
        high_ratio_df = pd.DataFrame({'label': problematic_labels})
        high_ratio_path = os.path.join(output_dir, f'{well}_high_ratio_segmentation_errors.parquet')
        high_ratio_df.to_parquet(high_ratio_path)
        logger.info(f"Saved high ratio segmentation errors filter: {high_ratio_path}")
