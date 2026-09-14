#!/usr/bin/env python3
"""
FAST VERSION: Wasserstein Distance Analysis with Progress Tracking

Optimizations:
1. Detailed progress tracking at every step
2. Option to cache combined weights locally
3. Real-time ETA estimates
4. Testing mode to run on subset of genes
"""

import pandas as pd
import numpy as np
import argparse
import yaml
import sys
import os
import time
from pathlib import Path
from typing import Sequence, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for progress bars: pip install tqdm")


# ============================================================================
# ARCHETYPE FILTERING & CLR TRANSFORMATION
# ============================================================================


def dirichlet_smooth(W: np.ndarray, alpha: float = 1e-6) -> np.ndarray:
    K = W.shape[1]
    return (W + alpha) / (1.0 + K * alpha)


def apply_clr_transform(W, eps: float = 1e-6):
    if isinstance(W, pd.DataFrame):
        index = W.index
        columns = W.columns
        W_vals = W.values
    else:
        index = None
        columns = None
        W_vals = W
    
    W_smooth = dirichlet_smooth(W_vals)
    L = np.log(np.clip(W_smooth, eps, None))
    clr_W = L - L.mean(axis=1, keepdims=True)
    
    if index is not None:
        return pd.DataFrame(clr_W, index=index, columns=columns)
    return clr_W


# ============================================================================
# WASSERSTEIN DISTANCE FUNCTIONS
# ============================================================================

def _weighted_median(x_sorted: np.ndarray, w: np.ndarray) -> float:
    cs = np.cumsum(w) / w.sum()
    idx = np.searchsorted(cs, 0.5, side="left")
    return x_sorted[min(idx, len(x_sorted) - 1)]


def _weighted_quantile(x_sorted: np.ndarray, w: np.ndarray, q: float) -> float:
    cs = np.cumsum(w) / w.sum()
    idx = np.searchsorted(cs, q, side="left")
    return x_sorted[min(idx, len(x_sorted) - 1)]


def wasserstein_1d_same_support(v: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """Vectorised W1 for two distributions on the identical sorted support v.

    Equivalent to wasserstein_1d(v, v, p, q, assume_sorted=True) but replaces
    the Python while-loop with two cumsum + a dot product, giving ~50-100x speedup.
    """
    p_norm = p / p.sum()
    q_norm = q / q.sum()
    gaps = np.diff(v)
    cdf_diff = np.abs(np.cumsum(p_norm)[:-1] - np.cumsum(q_norm)[:-1])
    return float(np.dot(cdf_diff, gaps))


def wasserstein_1d(
    x: np.ndarray, y: np.ndarray,
    wx: Optional[np.ndarray] = None, wy: Optional[np.ndarray] = None,
    assume_sorted: bool = False
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if wx is None: wx = np.ones_like(x)
    if wy is None: wy = np.ones_like(y)
    wx = np.asarray(wx, dtype=float)
    wy = np.asarray(wy, dtype=float)
    
    if not assume_sorted:
        ix = np.argsort(x); x = x[ix]; wx = wx[ix]
        iy = np.argsort(y); y = y[iy]; wy = wy[iy]
    
    wx = wx / wx.sum()
    wy = wy / wy.sum()

    i = j = 0
    rx = wx[0]; ry = wy[0]
    cost = 0.0
    nx, ny = len(x), len(y)

    while True:
        w = rx if rx < ry else ry
        cost += w * abs(x[i] - y[j])
        rx -= w; ry -= w
        if rx <= 0:
            i += 1
            if i == nx: break
            rx = wx[i]
        if ry <= 0:
            j += 1
            if j == ny: break
            ry = wy[j]
    return float(cost)


def trimmed_wasserstein_1d_on_support(
    support_sorted: np.ndarray,
    p_weights: np.ndarray,
    q_weights: np.ndarray,
    alpha: float = 0.02
) -> float:
    v = support_sorted
    p = p_weights.copy()
    q = q_weights.copy()
    p = p / (p.sum() + 1e-15)
    q = q / (q.sum() + 1e-15)

    q_lo_p = _weighted_quantile(v, p, alpha)
    q_hi_p = _weighted_quantile(v, p, 1 - alpha)
    q_lo_q = _weighted_quantile(v, q, alpha)
    q_hi_q = _weighted_quantile(v, q, 1 - alpha)

    p_trim = p.copy()
    q_trim = q.copy()
    p_trim[(v < q_lo_p) | (v > q_hi_p)] = 0.0
    q_trim[(v < q_lo_q) | (v > q_hi_q)] = 0.0
    
    if p_trim.sum() == 0 or q_trim.sum() == 0:
        return 0.0
    
    p_trim /= p_trim.sum()
    q_trim /= q_trim.sum()

    return wasserstein_1d_same_support(v, p_trim, q_trim)


def w1_with_stratified_permutation(
    values: np.ndarray,
    group: np.ndarray,
    strata: Optional[np.ndarray] = None,
    n_perm: int = 250,
    random_state: Optional[int] = 0,
    trimmed_alpha: Optional[float] = None
) -> Dict[str, float]:
    rng = np.random.default_rng(random_state)
    values = np.asarray(values, dtype=float)
    group = np.asarray(group, dtype=int)
    
    if strata is None:
        strata = np.zeros_like(group)
    else:
        strata = np.asarray(strata)

    order = np.argsort(values)
    v = values[order]
    g = group[order]
    s = strata[order]

    _, inv = np.unique(s, return_inverse=True)
    counts = np.bincount(inv)
    order_by_stratum = np.argsort(inv)
    starts = np.r_[0, counts.cumsum()[:-1]]
    ends = counts.cumsum()
    blocks = [order_by_stratum[starts[i]:ends[i]] for i in range(len(counts))]

    p_obs = g.astype(float)
    q_obs = 1.0 - p_obs
    
    if p_obs.sum() == 0 or q_obs.sum() == 0:
        return {"w1": 0.0, "pval": 1.0, "signed_w1": 0.0, "w1_trimmed": 0.0}

    w1_obs = wasserstein_1d_same_support(v, p_obs, q_obs)
    med_p = _weighted_median(v, p_obs)
    med_q = _weighted_median(v, q_obs)
    signed = float(np.sign(med_p - med_q) * w1_obs)

    w1_trim = None
    if trimmed_alpha is not None and trimmed_alpha > 0:
        w1_trim = trimmed_wasserstein_1d_on_support(v, p_obs, q_obs, trimmed_alpha)

    ge = 0
    g_perm = np.empty_like(g)
    for _ in range(n_perm):
        for blk in blocks:
            g_perm[blk] = rng.permutation(g[blk])
        p_perm = g_perm.astype(float)
        q_perm = 1.0 - p_perm
        if p_perm.sum() == 0 or q_perm.sum() == 0:
            continue
        w1_perm = wasserstein_1d_same_support(v, p_perm, q_perm)
        if w1_perm >= w1_obs:
            ge += 1
    
    pval = (ge + 1) / (n_perm + 1)

    out = {"w1": float(w1_obs), "pval": float(pval), "signed_w1": float(signed)}
    if w1_trim is not None:
        out["w1_trimmed"] = float(w1_trim)
    return out


def scan_archetypes_w1(
    Z: np.ndarray,
    group: np.ndarray,
    strata: Optional[np.ndarray] = None,
    n_perm: int = 250,
    trimmed_alpha: Optional[float] = 0.02,
    seed: Optional[int] = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    K = Z.shape[1]
    w1 = np.zeros(K, dtype=float)
    pv = np.ones(K, dtype=float)
    sw = np.zeros(K, dtype=float)
    wt = np.zeros(K, dtype=float) if trimmed_alpha not in (None, 0) else None

    for k in range(K):
        res = w1_with_stratified_permutation(
            Z[:, k], group, strata,
            n_perm=n_perm,
            random_state=None if seed is None else seed + k,
            trimmed_alpha=trimmed_alpha
        )
        w1[k] = res["w1"]
        pv[k] = res["pval"]
        sw[k] = res["signed_w1"]
        if wt is not None:
            wt[k] = res.get("w1_trimmed", np.nan)
    
    return w1, pv, sw, wt


def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    p = pvals.ravel().astype(float)
    finite = np.isfinite(p)
    m = finite.sum()
    q = np.full_like(p, np.nan, dtype=float)
    
    if m == 0:
        return q.reshape(pvals.shape)
    
    order = np.argsort(p[finite])
    ranked = p[finite][order]
    mult = m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate((ranked * mult)[::-1])[::-1]
    q_vals = np.minimum(q_sorted, 1.0)
    
    tmp = np.empty_like(q_vals)
    tmp[order] = q_vals
    q[finite] = tmp
    
    return q.reshape(pvals.shape)


def _process_single_gene(
    g: str,
    genes_arr: np.ndarray,
    wells_arr: np.ndarray,
    is_ctrl: np.ndarray,
    Z: np.ndarray,
    arch_cols: list,
    n_perm: int,
    trimmed_alpha: Optional[float],
    seed_offset: int
) -> Tuple[str, Optional[Dict]]:
    try:
        mask_g = (genes_arr == g)
        wells_g = np.unique(wells_arr[mask_g])
        mask_ctrl = is_ctrl & np.isin(wells_arr, wells_g)
        
        if not mask_ctrl.any() or not mask_g.any():
            return g, None
        
        use = mask_g | mask_ctrl
        group = mask_g[use].astype(np.int8)
        strata = wells_arr[use]
        Z_sub = Z[use, :]
        
        n_gene_val = int(group.sum())
        n_ctrl_val = int((~group.astype(bool)).sum())
        n_wells_val = int(wells_g.size)
        
        w1, pvals, signed_w1, w1_trim = scan_archetypes_w1(
            Z_sub, group, strata,
            n_perm=n_perm,
            trimmed_alpha=trimmed_alpha,
            seed=seed_offset
        )
        
        return g, {
            'w1': w1,
            'pvals': pvals,
            'signed_w1': signed_w1,
            'w1_trim': w1_trim,
            'n_gene': n_gene_val,
            'n_ctrl': n_ctrl_val,
            'n_wells': n_wells_val
        }
    except Exception as e:
        print(f"  Error processing gene {g}: {str(e)}")
        return g, None


# ============================================================================
# DATA LOADING
# ============================================================================

def load_archetype_weights(params_file: str, base_path: str, archetypes_to_remove: Optional[Dict] = None, channel_subset: Optional[Sequence[str]] = None) -> pd.DataFrame:
    print("\n" + "="*80)
    print("LOADING ARCHETYPE WEIGHTS")
    print("="*80)
    
    with open(params_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle both old format (list) and new format (dict with 'channels' key)
    if isinstance(config, list):
        params = config
        path_template = None
    else:
        params = config.get('channels', config)
        path_template = config.get('path_template', None)
        # Read archetypes_to_remove from config if not provided as argument
        if archetypes_to_remove is None:
            archetypes_to_remove = config.get('archetypes_to_remove', {})
    
    # Default path template if not specified
    if path_template is None:
        # path_template = "{base_path}/{channel}/k{k}_gamma{gamma}_knn{knn}_npc{npc}/archweights-{channel}_k{k}_gamma{gamma}_knn{knn}_npc{npc}.parquet"
        path_template = "{base_path}/{channel}/robust_archweights-{channel}_k{k}_gamma{gamma}_knn{knn}_npc{npc}.parquet"
    
    base_path = base_path.rstrip('/')

    if channel_subset is not None:
        available = [p['channel'] for p in params]
        missing = set(channel_subset) - set(available)
        if missing:
            raise ValueError(f"Requested channels not found in config: {sorted(missing)}")
        params = [p for p in params if p['channel'] in channel_subset]
        print(f"\n  Channel subset requested: {channel_subset}")

    print(f"\nLoaded parameters for {len(params)} channels")
    print(f"Path template: {path_template}")

    all_weights = []

    for param in params:
        channel = param['channel']
        k = param['k']
        gamma = param['gamma']
        knn = param['knn']
        npc = param['npc']
        
        print(f"\n  Loading {channel} (k={k}, gamma={gamma}, knn={knn}, npc={npc})...")
        
        s3_path = path_template.format(base_path=base_path, channel=channel, k=k, gamma=gamma, knn=knn, npc=npc)
        
        try:
            weights = pd.read_parquet(s3_path)
            weights.set_index(['Label', 'Guide', 'Gene', 'Well'], inplace=True)
            weights = weights.add_prefix(f'{channel}_')
            clr_weights = apply_clr_transform(weights)
            all_weights.append(clr_weights)
            print(f"    ✓ Loaded {clr_weights.shape[1]} archetypes, {clr_weights.shape[0]:,} cells")
        except Exception as e:
            print(f"    ✗ Error loading {channel}: {str(e)}")
            continue
    
    if all_weights:
        print(f"\nCombining weights from {len(all_weights)} channels...")
        combined_weights = pd.concat(all_weights, axis=1, join='outer')
        print(f"  Combined shape: {combined_weights.shape}")
    else:
        raise ValueError("No weights were successfully loaded!")
    
    if archetypes_to_remove:
        archetypes_to_drop = [item for sublist in archetypes_to_remove.values() for item in sublist]
        combined_weights_filtered = combined_weights.drop(columns=archetypes_to_drop, errors='ignore')
        n_dropped = combined_weights.shape[1] - combined_weights_filtered.shape[1]
        print(f"\n  Removed {n_dropped} technical artifact archetypes")
        print(f"  Categories: {list(archetypes_to_remove.keys())}")
    else:
        combined_weights_filtered = combined_weights
        print(f"\n  No archetypes to remove specified")
    
    print(f"  Final archetype count: {combined_weights_filtered.shape[1]}")
    print("\n" + "="*80 + "\n")
    
    return combined_weights_filtered


def load_or_cache_weights(params_file: str, base_path: str, cache_file: Optional[str] = None, archetypes_to_remove: Optional[Dict] = None, channel_subset: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if cache_file and os.path.exists(cache_file):
        print(f"\n{'='*80}")
        print(f"LOADING FROM CACHE: {cache_file}")
        print(f"{'='*80}")
        start = time.time()
        df = pd.read_parquet(cache_file)
        elapsed = time.time() - start
        print(f"\n✓ Loaded in {elapsed:.1f} seconds")
        print(f"  Shape: {df.shape}")
        if channel_subset is not None:
            keep = [c for c in df.columns if c.rsplit('_', 1)[0] in channel_subset]
            if not keep:
                raise ValueError(f"No columns matched channel subset {channel_subset}. "
                                 f"Available prefixes: {sorted(set(c.rsplit('_', 1)[0] for c in df.columns))}")
            df = df[keep]
            print(f"  Channel subset filter: {channel_subset}")
            print(f"  Columns after filter: {df.shape[1]}")
        print(f"  Archetypes: {df.shape[1]}")
        print(f"  Cells: {df.shape[0]:,}")
        return df
    
    print(f"\n{'='*80}")
    print("LOADING FROM S3 (this will take several minutes...)")
    print(f"{'='*80}")
    start = time.time()
    
    df = load_archetype_weights(params_file, base_path, archetypes_to_remove, channel_subset)
    
    elapsed = time.time() - start
    print(f"\n✓ Loaded in {elapsed/60:.1f} minutes")
    
    if cache_file:
        print(f"\nSaving cache to: {cache_file}")
        cache_path = Path(cache_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)
        print(f"✓ Cache saved (future runs will be faster!)")
    
    return df


def get_control_genes(df: pd.DataFrame, gene_level: str = "Gene") -> list:
    genes = df.index.get_level_values(gene_level)
    or_genes = genes[genes.str.startswith('OR', na=False)].unique().tolist()
    control_genes = or_genes + ['NTC']
    
    print(f"Control genes identified: {len(control_genes)} total")
    print(f"  - NTC: 1")
    print(f"  - OR genes: {len(or_genes)}")
    
    return control_genes


# ============================================================================
# PARALLEL ANALYSIS WITH PROGRESS
# ============================================================================

def gene_by_archetype_w1_tables_parallel(
    df: pd.DataFrame,
    arch_cols: Sequence[str],
    control_genes: Sequence[str] = ("NTC",),
    n_perm: int = 250,
    trimmed_alpha: Optional[float] = 0.02,
    seed: int = 0,
    n_jobs: int = -1,
    max_genes: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    print("\n" + "="*80)
    print("WASSERSTEIN DISTANCE ANALYSIS (PARALLEL WITH PROGRESS)")
    print("="*80)
    
    wells = df.index.get_level_values("Well").to_numpy()
    genes = df.index.get_level_values("Gene").to_numpy()
    Z = df.loc[:, arch_cols].to_numpy(dtype=np.float64)
    
    K = len(arch_cols)
    genes_arr = genes.astype(object)
    wells_arr = wells.astype(object)
    is_ctrl = np.isin(genes_arr, np.array(control_genes, dtype=object))
    test_genes = np.unique(genes_arr[~is_ctrl])
    
    if max_genes is not None and max_genes < len(test_genes):
        print(f"\n⚠️  LIMITING TO FIRST {max_genes} GENES (testing mode)")
        test_genes = test_genes[:max_genes]
    
    print(f"\nDataset info:")
    print(f"  Total cells: {len(df):,}")
    print(f"  Total archetypes: {K}")
    print(f"  Control genes: {len(control_genes)}")
    print(f"  Test genes: {len(test_genes)}")
    print(f"  Permutations: {n_perm}")
    
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    print(f"  Using {n_jobs} parallel workers")
    
    w1_mat = np.full((len(test_genes), K), np.nan, dtype=float)
    sw_mat = np.full((len(test_genes), K), np.nan, dtype=float)
    wt_mat = np.full((len(test_genes), K), np.nan, dtype=float) if trimmed_alpha not in (None, 0) else None
    pv_mat = np.full((len(test_genes), K), np.nan, dtype=float)
    n_gene = np.zeros(len(test_genes), dtype=int)
    n_ctrl = np.zeros(len(test_genes), dtype=int)
    n_wells_gene = np.zeros(len(test_genes), dtype=int)
    
    print(f"\nProcessing {len(test_genes)} genes in parallel...")
    print(f"Note: With {len(df):,} cells, first gene may take 30-90 seconds to complete.\n")
    
    start_time = time.time()
    last_update = start_time
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_idx = {}
        for i, g in enumerate(test_genes):
            future = executor.submit(
                _process_single_gene,
                g, genes_arr, wells_arr, is_ctrl, Z, arch_cols,
                n_perm, trimmed_alpha, seed + i
            )
            future_to_idx[future] = (i, g)
        
        # Progress tracking
        if HAS_TQDM:
            pbar = tqdm(total=len(test_genes), desc="Genes processed", unit="gene")
        else:
            pbar = None
            completed = 0
        
        for future in as_completed(future_to_idx):
            i, g = future_to_idx[future]
            gene_name, result = future.result()
            
            if result is not None:
                w1_mat[i, :] = result['w1']
                pv_mat[i, :] = result['pvals']
                sw_mat[i, :] = result['signed_w1']
                if wt_mat is not None:
                    wt_mat[i, :] = result['w1_trim']
                n_gene[i] = result['n_gene']
                n_ctrl[i] = result['n_ctrl']
                n_wells_gene[i] = result['n_wells']
            
            if pbar:
                pbar.update(1)
                if pbar.n == 1:
                    # First gene completed - give initial estimate
                    elapsed = time.time() - start_time
                    estimated_total = elapsed * len(test_genes) / n_jobs
                    print(f"\n  First gene completed in {elapsed:.0f}s - Estimated total: {estimated_total/60:.1f} min")
                if pbar.n % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = pbar.n / elapsed
                    remaining = (len(test_genes) - pbar.n) / rate
                    pbar.set_postfix({
                        'rate': f'{rate:.1f} genes/min',
                        'ETA': f'{remaining/60:.1f} min'
                    })
            else:
                completed += 1
                # Show first completion
                if completed == 1:
                    elapsed = time.time() - start_time
                    estimated_total = elapsed * len(test_genes) / n_jobs
                    print(f"  First gene completed in {elapsed:.0f}s - Estimated total: {estimated_total/60:.1f} min")
                # Regular updates
                if completed % 20 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    remaining = (len(test_genes) - completed) / rate
                    print(f"  Progress: {completed}/{len(test_genes)} ({100*completed/len(test_genes):.1f}%) - "
                          f"Rate: {rate:.1f} genes/min - ETA: {remaining/60:.1f} min")
        
        if pbar:
            pbar.close()
    
    elapsed = time.time() - start_time
    print(f"\n✓ All genes processed in {elapsed/60:.1f} minutes")
    print(f"  Average: {elapsed/len(test_genes):.1f} seconds per gene")
    
    print("\nBuilding output tables...")
    
    index = pd.Index(test_genes, name="Gene")
    w1_df = pd.DataFrame(w1_mat, index=index, columns=arch_cols)
    sw_df = pd.DataFrame(sw_mat, index=index, columns=arch_cols)
    pv_df = pd.DataFrame(pv_mat, index=index, columns=arch_cols)
    wt_df = pd.DataFrame(wt_mat, index=index, columns=arch_cols) if wt_mat is not None else None
    
    qvals = bh_qvalues(pv_df.to_numpy())
    q_df = pd.DataFrame(qvals, index=index, columns=arch_cols)
    
    long_df = pd.DataFrame({
        "Gene": np.repeat(test_genes, K),
        "Archetype": np.tile(arch_cols, len(test_genes)),
        "w1": w1_df.to_numpy().ravel(),
        "signed_w1": sw_df.to_numpy().ravel(),
        "pval": pv_df.to_numpy().ravel(),
        "qval": q_df.to_numpy().ravel(),
    }).astype({"Gene": str, "Archetype": str})
    
    if wt_df is not None:
        long_df["w1_trimmed"] = wt_df.to_numpy().ravel()
    
    counts_df = pd.DataFrame(
        {"n_cells_gene": n_gene, "n_cells_ctrl": n_ctrl, "n_wells_gene": n_wells_gene},
        index=index
    )
    
    n_sig = (q_df < 0.05).sum().sum()
    print(f"\nResults summary:")
    print(f"  Total tests: {len(test_genes) * K:,}")
    print(f"  Significant (q < 0.05): {n_sig:,} ({100*n_sig/(len(test_genes)*K):.2f}%)")
    print(f"  Genes with ≥1 significant archetype: {(q_df < 0.05).any(axis=1).sum()}")
    
    out = {
        "w1": w1_df,
        "signed_w1": sw_df,
        "pval": pv_df,
        "qval": q_df,
        "counts": counts_df,
        "long": long_df
    }
    if wt_df is not None:
        out["w1_trimmed"] = wt_df
    
    return out


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fast Wasserstein Distance Analysis with Caching',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--config', type=str, default='../../config/params-archetypes.yaml')
    parser.add_argument('--output', type=str, default='../../test_output/archetypes/wasserstein/')
    parser.add_argument('--s3_base', type=str, 
                       default='s3://bigdipir-ctg-s3/internal/singa166-lab/Carolina/cellmapp/archetypes_test')
    parser.add_argument('--n_perm', type=int, default=250)
    parser.add_argument('--trimmed_alpha', type=float, default=0.02)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cache', type=str, default=None,
                       help='Path to cache file (e.g., weights_cache.parquet)')
    parser.add_argument('--max_genes', type=int, default=None,
                       help='Limit to first N genes (for testing)')
    parser.add_argument('--channels', type=str, default=None,
                       help='Comma-separated list of channels to include (e.g., DAPI1,Phalloidin,LAMP1). '
                            'If not specified, all channels in the config are used.')
    
    args = parser.parse_args()
    
    is_s3 = args.output.startswith('s3://')
    if is_s3:
        output_dir = args.output.rstrip('/')
    else:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("FAST WASSERSTEIN PERTURBATION ANALYSIS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Config: {args.config}")
    print(f"  Output: {args.output}")
    print(f"  Permutations: {args.n_perm}")
    print(f"  Workers: {args.n_jobs if args.n_jobs > 0 else 'all cores'}")
    print(f"  Cache: {args.cache if args.cache else 'disabled'}")
    channel_subset = [c.strip() for c in args.channels.split(',')] if args.channels else None
    if channel_subset:
        print(f"  Channels: {channel_subset}")
    else:
        print(f"  Channels: all (from config)")
    if args.max_genes:
        print(f"  ⚠️  Max genes: {args.max_genes} (TESTING MODE)")
    
    # Load data with caching
    combined_weights = load_or_cache_weights(args.config, args.s3_base, args.cache, channel_subset=channel_subset)
    
    # Get control genes
    control_genes = get_control_genes(combined_weights)
    
    # Run analysis
    trimmed_alpha = args.trimmed_alpha if args.trimmed_alpha > 0 else None
    
    results = gene_by_archetype_w1_tables_parallel(
        df=combined_weights,
        arch_cols=list(combined_weights.columns),
        control_genes=control_genes,
        n_perm=args.n_perm,
        trimmed_alpha=trimmed_alpha,
        seed=args.seed,
        n_jobs=args.n_jobs,
        max_genes=args.max_genes
    )
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80 + "\n")

    def out_path(filename):
        if is_s3:
            return f"{output_dir}/{filename}"
        return str(output_dir / filename)

    results['w1'].to_parquet(out_path('wasserstein_w1_wide.parquet'))
    print("  ✓ wasserstein_w1_wide.parquet")

    results['signed_w1'].to_parquet(out_path('wasserstein_signed_w1_wide.parquet'))
    print("  ✓ wasserstein_signed_w1_wide.parquet")

    results['pval'].to_parquet(out_path('wasserstein_pval_wide.parquet'))
    print("  ✓ wasserstein_pval_wide.parquet")

    results['qval'].to_parquet(out_path('wasserstein_qval_wide.parquet'))
    print("  ✓ wasserstein_qval_wide.parquet")

    if 'w1_trimmed' in results:
        results['w1_trimmed'].to_parquet(out_path('wasserstein_w1_trimmed_wide.parquet'))
        print("  ✓ wasserstein_w1_trimmed_wide.parquet")

    results['counts'].to_parquet(out_path('gene_counts.parquet'))
    print("  ✓ gene_counts.parquet")

    results['long'].to_parquet(out_path('wasserstein_results_long.parquet'))
    print("  ✓ wasserstein_results_long.parquet")

    results['long'].to_csv(out_path('wasserstein_results_long.csv'), index=False)
    print("  ✓ wasserstein_results_long.csv")

    # Summary
    summary = results['long'].groupby('Gene').agg({
        'w1': ['mean', 'max'],
        'signed_w1': 'mean',
        'pval': 'min',
        'qval': 'min'
    }).round(6)
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.sort_values('qval_min')
    summary.to_csv(out_path('gene_summary.csv'))
    print("  ✓ gene_summary.csv")

    top_hits = results['long'][results['long']['qval'] < 0.05].sort_values('w1', ascending=False)
    top_hits.to_csv(out_path('significant_hits_qval05.csv'), index=False)
    print(f"  ✓ significant_hits_qval05.csv ({len(top_hits)} hits)")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
