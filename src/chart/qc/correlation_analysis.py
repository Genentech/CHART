"""
Correlation analysis functions for quality control.

This module contains functions for computing correlations between
different imaging rounds/cycles and applying quality control thresholds.
"""

import logging

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from .io import (
    SCALLOPS_AVAILABLE,
    load_pheno_dapi_images,
    load_pheno_to_sbs_dapi_image,
    load_sbs_dapi_image,
    load_sbs_objects
)
from .plotting import (
    plot_pheno_correlations,
    plot_pheno_correlation_filter,
    plot_pheno_sbs_correlations,
    plot_pheno_sbs_correlation_filter
)

logger = logging.getLogger(__name__)

# Bounding box columns, in scikit-image regionprops order:
# (min_row, min_col, max_row, max_col).  Rows are the y axis, columns the x
# axis.  These are selected by name because the upstream parquet does not
# store them in sorted order.
NUCLEI_BBOX_COLS = ['nuclei_bbox-0', 'nuclei_bbox-1', 'nuclei_bbox-2', 'nuclei_bbox-3']


def compute_pheno_correlations(objects, pheno_dir, well, cor_threshold=0.8, save_plots=False,
                               output_dir=None, *, dapi_channel_names):
    """Compute correlations between DAPI rounds in pheno images.
    Raises:
        RuntimeError: if scallops is unavailable, or if no object yielded a
            usable correlation (which indicates a configuration problem
            rather than poor data).
        ValueError: if fewer than 2 DAPI channels are given.
    """
    if not SCALLOPS_AVAILABLE:
        raise RuntimeError("scallops is required to compute pheno correlations but is not installed")

    n_rounds = len(dapi_channel_names or [])
    if n_rounds < 2:
        raise ValueError(f"Need at least 2 DAPI channels for correlation analysis, got {n_rounds}")

    logger.info(f"Loading registered pheno images ({n_rounds} DAPI rounds)...")
    if_reg = load_pheno_dapi_images(pheno_dir, well, dapi_channel_names)

    bbox = objects.loc[:, NUCLEI_BBOX_COLS]
    bbox = bbox.astype(pd.Int64Dtype())

    ref_ch = dapi_channel_names[0]
    other_chs = dapi_channel_names[1:]
    failures = []

    def cor_in_crop(row, img):
        """Correlate *ref_ch* against every other DAPI channel in a crop."""
        try:
            tmp = img.isel(x=slice(row['nuclei_bbox-1'], row['nuclei_bbox-3']),
                           y=slice(row['nuclei_bbox-0'], row['nuclei_bbox-2']))
            ref = tmp.loc[ref_ch].to_numpy().flatten()
            return [pearsonr(ref, tmp.loc[ch].to_numpy().flatten())[0]
                    for ch in other_chs]
        except Exception as e:
            if len(failures) < 3:
                failures.append(f"object {row.name}: {type(e).__name__}: {e}")
            return [np.nan] * len(other_chs)

    logger.info("Computing pheno correlations...")
    logger.info(f"Image channels available: {if_reg.c.values}")
    logger.info(f"Using DAPI channel names: {dapi_channel_names}")
    logger.info(f"Reference channel: {ref_ch} — comparing against {other_chs}")
    logger.info(f"Number of objects to process: {len(bbox)}")
    cors = bbox.apply(lambda x: cor_in_crop(x, img=if_reg), axis=1)
    cors = np.array(cors.tolist())
    logger.info(f"Correlation results shape: {cors.shape}")

    n_valid = int(np.sum(~np.isnan(cors)))
    logger.info(f"Number of valid correlations: {n_valid}")
    if n_valid == 0:
        detail = "; ".join(failures) if failures else "every correlation was NaN (constant image crops?)"
        raise RuntimeError(
            f"All {cors.shape[0]} pheno correlations failed for well {well}. "
            f"Check the DAPI channel names and the nuclei_bbox columns. First failures: {detail}"
        )
    if n_valid < cors.size:
        logger.warning(f"{cors.size - n_valid}/{cors.size} channel correlations could not be computed")
        if failures:
            logger.info(f"  First failures: {'; '.join(failures)}")

    # Compute mean correlations.  An object with any failed pair averages to
    # NaN and therefore fails the threshold
    mean_cors = np.mean(cors, axis=1)
    n_incomplete = int(np.sum(np.isnan(mean_cors)))
    if n_incomplete:
        logger.info(f"Objects excluded for having an incomplete correlation set: {n_incomplete}")
    
    # Plot mean correlations
    plot_pheno_correlations(objects, mean_cors, well, n_rounds, save_plots, output_dir)
    
    # Apply threshold filter
    pass_thre = mean_cors >= cor_threshold
    num_passing = pass_thre.sum()
    total_objects = len(pass_thre)
    logger.info(f"Nuclei passing pheno correlation threshold: {num_passing}")
    
    # Plot threshold results
    plot_pheno_correlation_filter(objects, pass_thre, well, save_plots, output_dir)
    
    return pass_thre, mean_cors, num_passing, total_objects


def compute_pheno_sbs_correlations(objects, pheno_to_sbs_dir, sbs_dir, sbs_feature_dir, well,
                                   cor_threshold=0.8, save_plots=False, output_dir=None,
                                   *, pheno_dapi_channel):
    """Compute correlations between pheno and SBS DAPI stains.

    *pheno_dapi_channel* selects the pheno DAPI channel, by name or by
    integer index; it comes from the ``pheno_sbs_dapi_channel`` setting.

    Raises:
        RuntimeError: if scallops is unavailable, or if no object yielded a
            usable correlation.
    """
    if not SCALLOPS_AVAILABLE:
        raise RuntimeError("scallops is required to compute pheno-SBS correlations but is not installed")
    
    logger.info("Loading registered pheno and SBS images...")
    
    pheno_reg = load_pheno_to_sbs_dapi_image(pheno_to_sbs_dir, well, pheno_dapi_channel)
    
    sbs = load_sbs_dapi_image(sbs_dir, well)
    
    # Load object data from registered space
    sbs_objects = load_sbs_objects(sbs_feature_dir, well, objects)
    
    # Extract bounding boxes
    sbsbbox = sbs_objects.loc[:, NUCLEI_BBOX_COLS]
    sbsbbox = sbsbbox.astype(pd.Int64Dtype())
    failures = []
    
    def cor_in_crop_pheno_sbs(row, pheno, sbs):
        """Compute correlation between pheno and SBS in a crop."""
        try:
            # Crop to bbox
            pheno_tmp = pheno.isel(x=slice(row['nuclei_bbox-1'], row['nuclei_bbox-3']),
                                   y=slice(row['nuclei_bbox-0'], row['nuclei_bbox-2']))
            sbs_tmp = sbs.isel(x=slice(row['nuclei_bbox-1'], row['nuclei_bbox-3']),
                               y=slice(row['nuclei_bbox-0'], row['nuclei_bbox-2']))
            
            # Compute correlation between images
            if len(pheno_tmp.data.flatten()) > 2 and len(sbs_tmp.data.flatten()) > 2:
                cor = pearsonr(pheno_tmp.data.flatten(), sbs_tmp.data.flatten())
                return cor[0]
            else:
                return np.nan
        except Exception as e:
            if len(failures) < 3:
                failures.append(f"object {row.name}: {type(e).__name__}: {e}")
            return np.nan
    
    logger.info("Computing pheno-SBS correlations...")
    sbs0 = sbs.isel(t=0)
    phenosbs_cors = sbsbbox.apply(lambda x: cor_in_crop_pheno_sbs(x, pheno=pheno_reg, sbs=sbs0), axis=1)

    n_valid = int(phenosbs_cors.notna().sum())
    logger.info(f"Number of valid correlations: {n_valid}/{len(phenosbs_cors)}")
    if n_valid == 0:
        detail = "; ".join(failures) if failures else "every correlation was NaN (constant image crops?)"
        raise RuntimeError(
            f"All {len(phenosbs_cors)} pheno-SBS correlations failed for well {well}. "
            f"Check the pheno DAPI channel and the nuclei_bbox columns. First failures: {detail}"
        )
    if n_valid < len(phenosbs_cors):
        logger.warning(f"{len(phenosbs_cors) - n_valid}/{len(phenosbs_cors)} correlations could not be computed")
        if failures:
            logger.info(f"  First failures: {'; '.join(failures)}")
    
    # Plot correlations
    plot_pheno_sbs_correlations(sbs_objects, phenosbs_cors, well, save_plots, output_dir)
    
    # Apply threshold filter
    pass_thre_phenosbs = phenosbs_cors >= cor_threshold
    num_passing_phenosbs = pass_thre_phenosbs.sum()
    total_objects_phenosbs = len(pass_thre_phenosbs)
    logger.info(f"Nuclei passing pheno-SBS correlation threshold: {num_passing_phenosbs}")
    
    # Plot threshold results
    plot_pheno_sbs_correlation_filter(sbs_objects, pass_thre_phenosbs, well, save_plots, output_dir)
    
    return pass_thre_phenosbs, phenosbs_cors, num_passing_phenosbs, total_objects_phenosbs
