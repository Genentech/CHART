"""
The effects stage: each gene against the controls on every archetype.

The statistics are tested on a small table of weights, and the run on
that table laid out where the archetype stage leaves its weights.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import wasserstein_distance

from chart.effects import COLUMNS, run_effects, workflow
from chart.io.config import EffectsConfig, load_archetypes_config

CHANNEL = 'CALR'
PARAMS = EffectsConfig(n_perm=20, n_jobs=1)


def weights():
    rng = np.random.default_rng(0)
    genes = np.repeat(['NTC', 'OR1A1', 'GENEA', 'GENEB', 'GENEC'], 40)
    table = pd.DataFrame(rng.dirichlet(np.ones(4), size=len(genes)),
                         columns=[f'Archetype{k}' for k in range(1, 5)])
    table['Well'] = np.where(genes == 'GENEC', 'C9',
                             np.tile(['A1', 'A2'], len(genes) // 2))
    table['Label'], table['Guide'], table['Gene'] = range(len(genes)), genes, genes
    return table


def test_the_fast_distance_is_the_wasserstein_distance():
    rng = np.random.default_rng(1)
    v, p, q = np.sort(rng.normal(size=50)), rng.random(50), rng.random(50)
    assert (workflow.wasserstein_1d_same_support(v, p, q)
            == pytest.approx(wasserstein_distance(v, v, p, q)))


def test_each_archetype_is_seeded_by_its_gene_and_position():
    table, _ = workflow.gene_effects(weights(), CHANNEL, params=PARAMS)
    data = weights()
    use = data.Gene.isin(['GENEB', 'NTC', 'OR1A1']).to_numpy()
    expected = workflow.w1_with_stratified_permutation(
        workflow.clr_transform(data.filter(like='Archetype').to_numpy())[use, 2],
        (data.Gene[use] == 'GENEB').to_numpy(), data.Well[use].to_numpy(),
        PARAMS.n_perm, PARAMS.seed + 1 + 2, PARAMS.trimmed_alpha)
    row = table[(table.Gene == 'GENEB') & (table.Archetype == 'CALR_Archetype3')]
    assert row.pval.item() == expected['pval']
    assert row.w1.item() == expected['w1']


def test_leaving_an_archetype_out_changes_nothing_else():
    full, skipped = workflow.gene_effects(weights(), CHANNEL, params=PARAMS)
    fewer, _ = workflow.gene_effects(weights(), CHANNEL, ['CALR_Archetype2'],
                                     PARAMS)
    pd.testing.assert_frame_equal(
        fewer, full[full.Archetype != 'CALR_Archetype2'].reset_index(drop=True))
    assert skipped == ['GENEC']


def test_an_archetype_the_channel_lacks_cannot_be_removed():
    with pytest.raises(ValueError, match='CALR_Archetype9'):
        workflow.gene_effects(weights(), CHANNEL, ['CALR_Archetype9'], PARAMS)


def config(tmp_path, **overrides):
    section = {'channels': [CHANNEL, 'FAK'], 'k': {CHANNEL: 4, 'FAK': 4}}
    section.update(overrides)
    return {'data_output_dir': str(tmp_path), 'local_output_dir': str(tmp_path),
            'archetypes': section, 'effects': {'n_perm': 20, 'n_jobs': 1}}


def with_npc_txt(tmp_path, **overrides):
    resolved = load_archetypes_config(config(tmp_path, **overrides))
    for channel in resolved.channels:
        directory = Path(resolved.channel_path(channel))
        directory.mkdir(parents=True)
        (directory / 'npc.txt').write_text('8\n')
    return resolved


def with_weights(tmp_path, **overrides):
    """Weights for CALR only, so FAK fails as a channel would before its
    weights step has run."""
    directory = Path(with_npc_txt(tmp_path, **overrides).channel_path(CHANNEL))
    weights().to_parquet(
        directory / f'robust_archweights-{CHANNEL}_k4_gamma30_knn10_npc8.parquet')
    return config(tmp_path, **overrides)


def test_missing_weights_are_reported_by_path(tmp_path):
    with pytest.raises(FileNotFoundError,
                       match=f'robust_archweights-{CHANNEL}_k4_gamma30_knn10_npc8'):
        with_npc_txt(tmp_path).weights_path(CHANNEL)


def test_a_configured_npc_must_agree_with_npc_txt(tmp_path):
    with pytest.raises(ValueError, match='npc.txt holds 8'):
        with_npc_txt(tmp_path, npc={CHANNEL: 12}).weights_path(CHANNEL)


def test_a_channel_gets_a_table_and_a_record_of_how_it_was_made(tmp_path):
    removal = {'debris': ['CALR_Archetype2', 'FAK_Archetype1']}
    outputs = run_effects(with_weights(tmp_path, archetypes_to_remove=removal),
                          channels=[CHANNEL])
    assert outputs == {CHANNEL: str(tmp_path / 'effects' / f'{CHANNEL}.parquet')}
    assert list(pd.read_parquet(outputs[CHANNEL]).columns) == COLUMNS
    record = json.loads((tmp_path / 'effects' / f'{CHANNEL}.json').read_text())
    assert record['parameters'] == {'n_perm': 20, 'trimmed_alpha': 0.02,
                                    'seed': 42}
    assert record['removed_archetypes'] == ['CALR_Archetype2']
    assert record['skipped_genes'] == ['GENEC']


def test_a_channel_without_weights_stops_the_run(tmp_path):
    with pytest.raises(FileNotFoundError, match='run the weights step'):
        run_effects(with_weights(tmp_path))


def test_a_channel_without_weights_can_be_passed_over(tmp_path):
    outputs = run_effects(with_weights(tmp_path), continue_on_error=True)
    assert outputs['FAK'] is None
    assert Path(outputs[CHANNEL]).exists()
