"""
Perturbation effects: how far each gene moves each archetype's weights
away from the controls.  The distance pools the control cells of every
well the gene appears in; the p-value shuffles the labels within each well.
Once every channel has its table, their p-values are corrected together.
"""

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .. import __version__
from ..dimred.channels import INDEX_COLUMNS
from ..dimred.pca import NoControlsError, control_mask
from ..io.config import (ARCHETYPES_SECTION, ArchetypesConfig, EffectsConfig,
                         SchemaConfig, join_path, load_archetypes_config,
                         load_effects_config)
from ..io.tables import load_table, save_table, table_exists, table_path

logger = logging.getLogger(__name__)

COLUMNS = ['Gene', 'Archetype', 'w1', 'signed_w1', 'w1_trimmed', 'pval',
           'n_gene', 'n_ctrl', 'n_wells']

POOLED = 'pooled'


def dirichlet_smooth(W: np.ndarray, alpha: float = 1e-6) -> np.ndarray:
    K = W.shape[1]
    return (W + alpha) / (1.0 + K * alpha)


def clr_transform(W: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Centred log-ratios of each row's weights, smoothed away from zero."""
    L = np.log(np.clip(dirichlet_smooth(W), eps, None))
    return L - L.mean(axis=1, keepdims=True)


def _weighted_quantile(x_sorted: np.ndarray, w: np.ndarray, q: float) -> float:
    cs = np.cumsum(w) / w.sum()
    idx = np.searchsorted(cs, q, side="left")
    return x_sorted[min(idx, len(x_sorted) - 1)]


def wasserstein_1d_same_support(v: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """W1 between two weightings of the same sorted support *v*."""
    p_norm = p / p.sum()
    q_norm = q / q.sum()
    gaps = np.diff(v)
    cdf_diff = np.abs(np.cumsum(p_norm)[:-1] - np.cumsum(q_norm)[:-1])
    return float(np.dot(cdf_diff, gaps))


def trimmed_wasserstein_1d_on_support(v: np.ndarray, p_weights: np.ndarray,
                                      q_weights: np.ndarray,
                                      alpha: float = 0.02) -> float:
    """W1 once each side's outer *alpha* tails are dropped."""
    p = p_weights / (p_weights.sum() + 1e-15)
    q = q_weights / (q_weights.sum() + 1e-15)

    p_trim = p.copy()
    q_trim = q.copy()
    p_trim[(v < _weighted_quantile(v, p, alpha))
           | (v > _weighted_quantile(v, p, 1 - alpha))] = 0.0
    q_trim[(v < _weighted_quantile(v, q, alpha))
           | (v > _weighted_quantile(v, q, 1 - alpha))] = 0.0

    if p_trim.sum() == 0 or q_trim.sum() == 0:
        return 0.0
    p_trim /= p_trim.sum()
    q_trim /= q_trim.sum()
    return wasserstein_1d_same_support(v, p_trim, q_trim)


def w1_with_stratified_permutation(values: np.ndarray, group: np.ndarray,
                                   strata: np.ndarray, n_perm: int = 1000,
                                   random_state: int = 0,
                                   trimmed_alpha: Optional[float] = None
                                   ) -> Dict[str, float]:
    """W1 between the two groups, with a p-value from shuffling the group
    labels within each stratum."""
    rng = np.random.default_rng(random_state)
    values = np.asarray(values, dtype=float)
    group = np.asarray(group, dtype=int)
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
    med_p = _weighted_quantile(v, p_obs, 0.5)
    med_q = _weighted_quantile(v, q_obs, 0.5)
    out = {"w1": float(w1_obs),
           "signed_w1": float(np.sign(med_p - med_q) * w1_obs)}
    if trimmed_alpha:
        out["w1_trimmed"] = float(
            trimmed_wasserstein_1d_on_support(v, p_obs, q_obs, trimmed_alpha))

    ge = 0
    g_perm = np.empty_like(g)
    for _ in range(n_perm):
        for blk in blocks:
            g_perm[blk] = rng.permutation(g[blk])
        p_perm = g_perm.astype(float)
        q_perm = 1.0 - p_perm
        if p_perm.sum() == 0 or q_perm.sum() == 0:
            continue
        if wasserstein_1d_same_support(v, p_perm, q_perm) >= w1_obs:
            ge += 1
    out["pval"] = float((ge + 1) / (n_perm + 1))
    return out


def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values, NaN wherever the p-value is not finite."""
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


_shared: Dict[str, np.ndarray] = {}


def _share(genes: np.ndarray, wells: np.ndarray, controls: np.ndarray,
           Z: np.ndarray) -> None:
    _shared.update(genes=genes, wells=wells, controls=controls, Z=Z)


def _test_gene(gene: str, seed: int, positions: Sequence[int], n_perm: int,
               trimmed_alpha: Optional[float]) -> Optional[List[Dict]]:
    """One gene against the pooled controls of the wells it appears in, on
    each archetype in *positions*; ``None`` if those wells hold none."""
    genes, wells, controls, Z = (_shared[key] for key in
                                 ('genes', 'wells', 'controls', 'Z'))
    in_gene = genes == gene
    gene_wells = np.unique(wells[in_gene])
    in_ctrl = controls & np.isin(wells, gene_wells)
    if not in_ctrl.any():
        return None

    use = in_gene | in_ctrl
    group = in_gene[use].astype(np.int8)
    strata = wells[use]
    counts = {'n_gene': int(group.sum()), 'n_ctrl': int((group == 0).sum()),
              'n_wells': int(gene_wells.size)}
    return [{**w1_with_stratified_permutation(Z[use, k], group, strata,
                                              n_perm, seed + k, trimmed_alpha),
             **counts}
            for k in positions]


def gene_effects(weights: pd.DataFrame, channel: str,
                 removed: Sequence[str] = (),
                 params: Optional[EffectsConfig] = None,
                 schema: Optional[SchemaConfig] = None
                 ) -> Tuple[pd.DataFrame, List[str]]:
    """Test every gene against the pooled controls of the wells it appears
    in, on each archetype.

    Args:
        weights: One channel's weights, as the weights step wrote them
        channel: Which channel, for naming the archetypes
        removed: Archetypes to leave out, as ``<channel>_ArchetypeN``

    Returns:
        One row per gene and archetype, and the genes left untested
        because their wells hold no controls.
    """
    params = params or EffectsConfig()
    missing = [name for name in INDEX_COLUMNS if name not in weights.columns]
    if missing:
        raise ValueError(f"The weights for {channel} have no "
                         f"{' or '.join(missing)} column")

    columns = [c for c in weights.columns if c not in INDEX_COLUMNS]
    archetypes = [f'{channel}_{c}' for c in columns]
    unknown = sorted(set(removed) - set(archetypes))
    if unknown:
        raise ValueError(
            f"'{ARCHETYPES_SECTION}.archetypes_to_remove' names "
            f"{', '.join(unknown)}, but {channel} has only "
            f"{', '.join(archetypes)}")
    positions = [k for k, name in enumerate(archetypes) if name not in removed]

    genes = weights['Gene'].to_numpy().astype(object)
    wells = weights['Well'].to_numpy().astype(object)
    controls = control_mask(weights['Gene'], schema)
    if not controls.any():
        raise NoControlsError(f"None of the {len(weights)} cells for "
                              f"{channel} carry a control guide")
    Z = clr_transform(weights[columns].to_numpy(dtype=np.float64))

    test_genes = np.unique(genes[~controls])
    n_jobs = os.cpu_count() if params.n_jobs == -1 else params.n_jobs
    logger.info(f"{channel}: testing {len(test_genes)} genes on "
                f"{len(positions)} of {len(archetypes)} archetypes, "
                f"{params.n_perm} permutations, {n_jobs} workers")

    rows, skipped = [], []
    step = max(1, len(test_genes) // 10)
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_share,
                             initargs=(genes, wells, controls, Z)) as pool:
        futures = [pool.submit(_test_gene, gene, params.seed + i, positions,
                               params.n_perm, params.trimmed_alpha)
                   for i, gene in enumerate(test_genes)]
        for i, (gene, future) in enumerate(zip(test_genes, futures)):
            try:
                result = future.result()
            except Exception as error:
                pool.shutdown(cancel_futures=True)
                raise RuntimeError(f"The effects test failed for gene {gene} "
                                   f"in {channel}: {error}") from error
            if result is None:
                skipped.append(str(gene))
            else:
                rows += [{'Gene': str(gene), 'Archetype': archetypes[k], **res}
                         for k, res in zip(positions, result)]
            if (i + 1) % step == 0:
                logger.info(f"{channel}: {i + 1}/{len(test_genes)} genes")

    if skipped:
        logger.warning(f"{channel}: {len(skipped)} genes have no controls in "
                       f"their wells and were not tested")
    return pd.DataFrame(rows, columns=COLUMNS), skipped


def write_channel_effects(archetypes: ArchetypesConfig,
                          effects: EffectsConfig, channel: str) -> str:
    """Test *channel*'s weights and write ``<channel>.parquet`` with a
    ``<channel>.json`` record beside it."""
    path = archetypes.weights_path(channel)
    removed = sorted({name for names in archetypes.archetypes_to_remove.values()
                      for name in names if name.rsplit('_', 1)[0] == channel})
    if not removed:
        logger.warning(f"Nothing in '{ARCHETYPES_SECTION}.archetypes_to_remove' "
                       f"belongs to {channel}, so all of its archetypes are "
                       f"tested, including any that are segmentation errors "
                       f"or staining artefacts. Those are best listed there "
                       f"before running effects.")
    logger.info(f"Reading {path}")
    table, skipped = gene_effects(pd.read_parquet(path), channel, removed,
                                  effects, effects.schema)

    directory = effects.outputs_path
    output = save_table(table, directory, channel)
    record = {'created': datetime.now().isoformat(timespec='seconds'),
              'chart_version': __version__, 'weights': path,
              'parameters': {key: getattr(effects, key)
                             for key in ('n_perm', 'trimmed_alpha', 'seed')},
              'removed_archetypes': removed, 'skipped_genes': skipped}
    _write_record(join_path(directory, f'{channel}.json'), record)
    return output


def _write_record(path: str, record: Dict[str, Any]) -> None:
    if '://' in path:
        import fsspec
        opener = fsspec.open
    else:
        opener = open
    with opener(path, 'w') as handle:
        json.dump(record, handle, indent=2)


def pool_effects(archetypes: ArchetypesConfig, effects: EffectsConfig,
                 failed: Sequence[str] = ()) -> Optional[str]:
    """Correct the p-values of every channel's table together, and write
    them with their q-values to ``pooled.parquet``, with a ``pooled.json``
    record beside it.

    A q-value depends on every test corrected alongside it, so nothing is
    written unless every channel in the config has a table from this run
    or an earlier one.

    Args:
        failed: Channels that failed in this run, so that any table of
                theirs on disk is an earlier run's

    Returns:
        The pooled table, or ``None`` if it was not written.
    """
    directory = effects.outputs_path
    channels = list(archetypes.channels)
    stale = [channel for channel in channels if channel in failed]
    missing = [channel for channel in channels if channel not in stale
               and not table_exists(directory, channel)]
    if stale or missing:
        if stale:
            reason = (f"{', '.join(stale)} failed in this run, and any table "
                      f"of theirs is from an earlier one")
        else:
            reason = (f"there is no effects table yet for "
                      f"{', '.join(missing)}, and the q-values are pooled "
                      f"only once every channel in "
                      f"'{ARCHETYPES_SECTION}.channels' has one")
        earlier = (f" The {POOLED}.parquet already in {directory} is from an "
                   f"earlier run." if table_exists(directory, POOLED) else '')
        logger.warning(f"Not pooling the q-values: {reason}.{earlier}")
        return None

    pooled = pd.concat([load_table(directory, channel) for channel in channels],
                       ignore_index=True)
    pooled['qval'] = bh_qvalues(pooled['pval'].to_numpy())
    output = save_table(pooled, directory, POOLED)
    record = {'created': datetime.now().isoformat(timespec='seconds'),
              'chart_version': __version__, 'method': 'Benjamini-Hochberg',
              'tests': len(pooled),
              'tables': [table_path(directory, channel) for channel in channels]}
    _write_record(join_path(directory, f'{POOLED}.json'), record)
    return output


def run_effects(config: Union[str, Dict[str, Any]],
                channels: Optional[Sequence[str]] = None,
                continue_on_error: bool = False) -> Dict[str, Optional[str]]:
    """Test each channel's archetype weights, then correct every channel's
    p-values together.

    Args:
        config: A configuration file or mapping; the channels, weights
                and removal list come from its archetypes section
        channels: Channels to run; ``None`` runs every channel in the config
        continue_on_error: Log and carry on when a channel fails

    Returns:
        Each channel's effects table, or ``None`` where it failed.
    """
    archetypes = load_archetypes_config(config)
    effects = load_effects_config(config)
    channels = list(channels) if channels else list(archetypes.channels)
    if not channels:
        raise ValueError(f"No channels to run: name them with --channel or "
                         f"under '{ARCHETYPES_SECTION}.channels'")

    outputs: Dict[str, Optional[str]] = {}
    for channel in channels:
        try:
            outputs[channel] = write_channel_effects(archetypes, effects,
                                                     channel)
        except Exception as error:
            if not continue_on_error:
                raise
            outputs[channel] = None
            logger.error(f"Effects failed for channel {channel}: {error}")

    failed = [channel for channel, output in outputs.items() if output is None]
    logger.info(f"Effects complete: {len(outputs) - len(failed)}/"
                f"{len(outputs)} channels succeeded")
    unlisted = [channel for channel in outputs
                if channel not in archetypes.channels]
    if unlisted:
        logger.warning(f"The pooled q-values cover only the channels in "
                       f"'{ARCHETYPES_SECTION}.channels', which does not list "
                       f"{', '.join(unlisted)}.")
    pool_effects(archetypes, effects, failed)
    return outputs
