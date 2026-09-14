"""
Segmentation error detection for quality control analysis.

This module contains functions for detecting segmentation errors based on
cytosol-to-nucleus area ratios.
"""

import logging
from typing import Optional

import pandas as pd

from .plotting import plot_segmentation_error_distribution

logger = logging.getLogger(__name__)


def detect_segmentation_errors(objects: pd.DataFrame, 
                              well: str,
                              ratio_threshold: float = 15.0,
                              save_plots: bool = False,
                              output_dir: Optional[str] = None) -> pd.Series:
    """
    Detect segmentation errors based on cytosol-to-nucleus area ratio.
    
    Args:
        objects: DataFrame containing object features with 'cytosol_area' and 'nuclei_area' columns
        well: Well identifier for labeling
        ratio_threshold: Threshold for cytosol/nucleus area ratio (default: 15.0)
        save_plots: Whether to save plots to output directory
        output_dir: Directory to save plots (required if save_plots=True)
    
    Returns:
        Boolean Series indicating which objects are likely segmentation errors
    """
    # Calculate cytosol-to-nucleus area ratio
    cyto_nuc_ratio = objects['cytosol_area'] / objects['nuclei_area']
    
    # Identify potential segmentation errors (high ratio)
    segmentation_errors = cyto_nuc_ratio > ratio_threshold
    
    # Log results
    n_errors = segmentation_errors.sum()
    total_objects = len(objects)
    logger.info(f"Segmentation error detection for well {well}:")
    logger.info(f"  Objects with high cytosol/nucleus ratio (>={ratio_threshold}): {n_errors}/{total_objects} ({n_errors/total_objects*100:.1f}%)")
    
    # Create visualization
    if save_plots and output_dir:
        plot_segmentation_error_distribution(cyto_nuc_ratio, well, ratio_threshold, output_dir)
    
    return segmentation_errors
