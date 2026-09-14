"""
Plotting functions for quality control analysis.

This module contains functions for creating visualizations
of object size distributions and other QC metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os


def plot_nuclear_size(objects, well, save_plots=False, output_dir=None):
    """Plot nuclear size distribution."""
    nuc_area = objects['nuclei_area']
    
    fig, ax = plt.subplots(figsize=(8.5, 7))
    plt.gca().invert_yaxis()
    scatter = plt.scatter(x=objects.loc[:, 'nuclei_centroid-1'],
                         y=objects.loc[:, 'nuclei_centroid-0'],
                         c=np.log10(nuc_area), s=0.01)
    plt.title(f'Nuclear size - {well}')
    plt.colorbar(scatter)
    
    if save_plots and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{well}_nuclear_size.png'), dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close(fig)


def plot_cell_size(objects, well, save_plots=False, output_dir=None):
    """Plot cell size distribution."""
    cell_area = objects['cell_area']
    
    fig, ax = plt.subplots(figsize=(8.5, 7))
    plt.gca().invert_yaxis()
    scatter = plt.scatter(x=objects.loc[:, 'cell_centroid-1'],
                         y=objects.loc[:, 'cell_centroid-0'],
                         c=np.log10(cell_area), s=0.01)
    plt.title(f'Cell size - {well}')
    plt.colorbar(scatter)
    
    if save_plots and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{well}_cell_size.png'), dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close(fig)


def plot_segmentation_error_distribution(cyto_nuc_ratio: pd.Series,
                                       well: str,
                                       ratio_threshold: float,
                                       output_dir: str) -> None:
    """
    Plot the distribution of cytosol-to-nucleus area ratios.
    
    Args:
        cyto_nuc_ratio: Series of cytosol/nucleus area ratios
        well: Well identifier for labeling
        ratio_threshold: Threshold line to draw
        output_dir: Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot of ratios vs nuclear area (log scale)
    scatter = ax.scatter(x=np.log1p(cyto_nuc_ratio.index),  # Using index as x-axis
                        y=cyto_nuc_ratio,
                        s=0.1, alpha=0.6)
    
    # Add threshold line
    ax.axhline(y=ratio_threshold, color='red', linestyle='--', 
               label=f'Threshold = {ratio_threshold}')
    
    ax.set_xlabel('Object Index')
    ax.set_ylabel('Cytosol/Nucleus Area Ratio')
    ax.set_title(f'Segmentation Error Detection - {well}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(os.path.join(output_dir, f'{well}_segmentation_error_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_pheno_correlations(objects, mean_cors, well, n_rounds,
                            save_plots=False, output_dir=None):
    """Plot mean DAPI correlation across pheno rounds."""
    fig, ax = plt.subplots(figsize=(8.5, 7))
    plt.gca().invert_yaxis()
    scatter = plt.scatter(x=objects.loc[:, 'nuclei_centroid-1'],
                         y=objects.loc[:, 'nuclei_centroid-0'],
                         s=0.01, c=mean_cors, cmap='RdYlBu_r',
                         vmin=-1, vmax=1)
    plt.title(f'Mean DAPI correlation across {n_rounds} pheno rounds - {well}')
    plt.colorbar(scatter)
    
    if save_plots and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{well}_pheno_correlations.png'), dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close(fig)


def plot_pheno_correlation_filter(objects, pass_thre, well,
                                  save_plots=False, output_dir=None):
    """Plot which nuclei pass the pheno correlation threshold."""
    cmap = mcolors.ListedColormap(['#AA98A9', '#007BA7'])
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.gca().invert_yaxis()
    plt.scatter(x=objects.loc[:, 'nuclei_centroid-1'],
                y=objects.loc[:, 'nuclei_centroid-0'],
                c=pass_thre, cmap=cmap, s=0.01)
    plt.title(f'Nuclei w/ good enough pheno registration - {well}')
    
    if save_plots and output_dir:
        plt.savefig(os.path.join(output_dir, f'{well}_pheno_correlation_filter.png'), dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close(fig)


def plot_pheno_sbs_correlations(sbs_objects, phenosbs_cors, well,
                                save_plots=False, output_dir=None):
    """Plot DAPI correlation between pheno and SBS."""
    fig, ax = plt.subplots(figsize=(8.5, 7))
    plt.gca().invert_yaxis()
    scatter = plt.scatter(x=sbs_objects.loc[:, 'nuclei_centroid-1'],
                         y=sbs_objects.loc[:, 'nuclei_centroid-0'],
                         s=0.01, c=phenosbs_cors, cmap='RdYlBu_r',
                         vmin=-1, vmax=1)
    plt.title(f'DAPI correlation between pheno and SBS - {well}')
    plt.colorbar(scatter)
    
    if save_plots and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{well}_pheno_sbs_correlations.png'), dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close(fig)


def plot_pheno_sbs_correlation_filter(sbs_objects, pass_thre_phenosbs, well,
                                      save_plots=False, output_dir=None):
    """Plot which nuclei pass the pheno-to-SBS correlation threshold."""
    cmap = mcolors.ListedColormap(['#AA98A9', '#007BA7'])
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.gca().invert_yaxis()
    plt.scatter(x=sbs_objects.loc[:, 'nuclei_centroid-1'],
                y=sbs_objects.loc[:, 'nuclei_centroid-0'],
                c=pass_thre_phenosbs, cmap=cmap, s=0.01)
    plt.title(f'Nuclei w/ good enough pheno-to-SBS registration - {well}')
    
    if save_plots and output_dir:
        plt.savefig(os.path.join(output_dir, f'{well}_pheno_sbs_correlation_filter.png'), dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close(fig)

