"""
The programs stage: sparse PCA on the pooled effects, modules of genes and
each cell's scores.

Tested on made-up effects of thirty genes in three groups, each group
moving its own archetypes, laid out where the effects stage leaves them.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.decomposition import SparsePCA

from chart.dimred.channels import INDEX_COLUMNS
from chart.io.config import load_archetypes_config
from chart.programs import STEPS, run_programs, workflow

CHANNELS = ('CALR', 'FAK')
ARCHETYPES = [f'{channel}_Archetype{k}' for channel in CHANNELS for k in range(1, 5)]
GROUPS = {'A': ['CALR_Archetype1', 'CALR_Archetype2'],
          'B': ['FAK_Archetype1', 'FAK_Archetype2'],
          'C': ['CALR_Archetype4', 'FAK_Archetype3']}
BBOX = ['cell_bbox-0', 'cell_bbox-1', 'cell_bbox-2', 'cell_bbox-3']


def pooled():
    """CALR_Archetype3, FAK_Archetype4 and the gene QUIET have no
    significant effect."""
    rng = np.random.default_rng(0)
    rows = []
    for group, moved in GROUPS.items():
        for i in range(10):
            for archetype in ARCHETYPES:
                effect = rng.normal(0, 0.05) + (rng.uniform(1, 2) if archetype in moved else 0)
                rows.append({'Gene': f'{group}{i}', 'Archetype': archetype,
                             'signed_w1': effect, 'qval': 0.001 if effect > 0.5 else 0.5})
    rows += [{'Gene': 'QUIET', 'Archetype': archetype,
              'signed_w1': rng.normal(0, 0.05), 'qval': 0.5} for archetype in ARCHETYPES]
    return pd.DataFrame(rows)


def matrix():
    return workflow.effect_matrix(pooled(), 0.01)


def fewer_refits(monkeypatch):
    monkeypatch.setattr(workflow, 'N_SEEDS', 2)
    monkeypatch.setattr(workflow, 'RUNS_PER_SEED', 2)


def cell_weights():
    rng = np.random.default_rng(1)
    tables = {}
    for channel in CHANNELS:
        table = pd.DataFrame(rng.dirichlet(np.ones(4), size=50),
                             columns=[f'Archetype{k}' for k in range(1, 5)])
        table['Well'], table['Label'], table['Guide'], table['Gene'] = 'A1', range(50), 'g1', 'NTC'
        tables[channel] = table
    return tables


def objects(labels):
    return pd.DataFrame({name: np.arange(len(labels)) + i for i, name in enumerate(BBOX)},
                        index=pd.MultiIndex.from_arrays(
                            [labels, ['g1'] * len(labels), ['NTC'] * len(labels),
                             ['A1'] * len(labels)],
                            names=['Label', 'Guide', 'Gene', 'Well']))


def test_the_matrix_keeps_genes_and_archetypes_with_any_significant_effect():
    data = matrix()
    assert list(data.index) == sorted(f'{g}{i}' for g in GROUPS for i in range(10))
    assert list(data.columns) == sorted(sum(GROUPS.values(), []))
    effects = pooled()
    assert data.loc['A0', 'CALR_Archetype1'] == effects.signed_w1[
        (effects.Gene == 'A0') & (effects.Archetype == 'CALR_Archetype1')].item()


def test_a_gene_untested_on_a_kept_archetype_is_reported():
    effects = pooled()
    effects = effects[~((effects.Gene == 'B3') & (effects.Archetype == 'CALR_Archetype1'))]
    with pytest.raises(ValueError, match='the first being B3'):
        workflow.effect_matrix(effects, 0.01)


def test_each_program_points_its_largest_loading_up():
    loadings, _ = workflow.fit_programs(matrix(), 3, 42)
    L = loadings.to_numpy()
    assert (L[np.abs(L).argmax(axis=0), np.arange(3)] > 0).all()


def test_gene_scores_are_sparse_pcas_with_the_average_gene_added_back():
    data = matrix()
    loadings, scores = workflow.fit_programs(data, 3, 42)
    model = SparsePCA(n_components=3, random_state=42, max_iter=500,
                      tol=1e-4).fit(data.to_numpy())
    signs = np.sign((loadings.to_numpy() * model.components_.T).sum(axis=0))
    np.testing.assert_allclose(scores, model.transform(data.to_numpy() + model.mean_) * signs,
                               atol=1e-10)


def test_genes_moving_the_same_archetypes_share_a_module(monkeypatch):
    fewer_refits(monkeypatch)
    data = matrix()
    loadings, _ = workflow.fit_programs(data, 3, 42)
    consensus, modules = workflow.gene_modules(data, loadings, 3, 42)
    assert (modules.groupby(modules.index.str[0])['module'].nunique() == 1).all()
    assert modules['module'].nunique() == 3
    np.testing.assert_array_equal(consensus, consensus.T)
    assert (np.diag(consensus) == 1).all()
    assert set(modules['assignment']) <= {'core', 'weak'}


def test_cells_are_scored_on_the_log_ratios_of_their_weights():
    loadings, _ = workflow.fit_programs(matrix(), 3, 42)
    weights = cell_weights()
    calr = workflow.clr_transform(weights['CALR'].filter(like='Archetype').to_numpy())
    weights['FAK'] = weights['FAK'].iloc[1:]
    scores = workflow.cell_scores(weights, loadings)

    first = pd.Series(0.0, index=loadings.index)
    first[['CALR_Archetype1', 'CALR_Archetype2', 'CALR_Archetype4']] = calr[0, [0, 1, 3]]
    np.testing.assert_allclose(scores.xs(0, level='Label'),
                               workflow.project(first.to_frame().T, loadings))


def test_weights_lacking_an_archetype_the_programs_load_on_are_reported():
    loadings, _ = workflow.fit_programs(matrix(), 3, 42)
    weights = cell_weights()
    del weights['FAK']
    with pytest.raises(ValueError, match='FAK_Archetype1'):
        workflow.cell_scores(weights, loadings)


def test_example_cells_are_the_purest_of_the_top_scorers(monkeypatch):
    monkeypatch.setattr(workflow, 'TOP_N_CELLS', 3)
    monkeypatch.setattr(workflow, 'TOP_N_PURE', 1)
    labels = list(range(10))
    scores = pd.DataFrame({'P1': [10, 9, 8] + [0] * 7, 'P2': [10, 0, 0] + list(range(1, 8))},
                          index=pd.MultiIndex.from_arrays(
                              [['A1'] * 10, labels, ['g1'] * 10, ['NTC'] * 10],
                              names=INDEX_COLUMNS), dtype=float)
    cells = workflow.example_cells(scores, objects(labels), BBOX)
    assert cells.set_index('Program').Label.to_dict() == {'P1': 1, 'P2': 9}
    assert cells[BBOX].notna().all().all()


def laid_out(tmp_path):
    """Pooled effects, both channels' weights and the objects table, where
    the earlier stages leave them."""
    settings = {'data_output_dir': str(tmp_path), 'local_output_dir': str(tmp_path),
                'archetypes': {'channels': list(CHANNELS), 'k': {c: 4 for c in CHANNELS}},
                'programs': {'k': 3, 'n_modules': 3}}
    (tmp_path / 'effects').mkdir()
    pooled().to_parquet(tmp_path / 'effects' / 'pooled.parquet')
    archetypes = load_archetypes_config(settings)
    for channel, table in cell_weights().items():
        directory = Path(archetypes.channel_path(channel))
        directory.mkdir(parents=True)
        (directory / 'npc.txt').write_text('8\n')
        table.to_parquet(directory / f'robust_archweights-{channel}_k4_gamma30_knn10_npc8.parquet')
    combined = tmp_path / 'preprocessing' / 'allwells' / 'combined'
    combined.mkdir(parents=True)
    objects(list(range(50))).to_parquet(combined / 'allwells-objects.parquet')
    return settings


def test_a_run_writes_every_table_with_a_record(tmp_path, monkeypatch):
    fewer_refits(monkeypatch)
    outputs = run_programs(laid_out(tmp_path))
    directory = tmp_path / 'programs'
    assert outputs == {step: str(directory / f'{name}.parquet') for step, name in
                       zip(STEPS, ('matrix', 'scores', 'modules', 'cell_scores'))}
    assert json.loads((directory / 'scores.json').read_text())['k'] == 3
    assert len(pd.read_parquet(directory / 'consensus.parquet')) == 30
    assert list(pd.read_parquet(directory / 'loadings.parquet').columns) == ['P1', 'P2', 'P3']
    top = pd.read_parquet(directory / 'top_cells.parquet')
    assert top.groupby('Program').size().to_dict() == {'P1': 20, 'P2': 20, 'P3': 20}


def test_a_step_without_the_one_before_it_says_which_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match='run the matrix step'):
        run_programs(laid_out(tmp_path), steps=['fit'])


def test_cells_refuse_programs_on_an_archetype_removed_since(tmp_path):
    settings = laid_out(tmp_path)
    run_programs(settings, steps=['matrix', 'fit'])
    settings['archetypes']['archetypes_to_remove'] = {'seg_err': ['FAK_Archetype1']}
    with pytest.raises(ValueError, match="FAK_Archetype1, which 'archetypes.archetypes_to_remove'"):
        run_programs(settings, steps=['cells'])


def test_the_example_notebook_runs_on_a_runs_tables(tmp_path, monkeypatch):
    fewer_refits(monkeypatch)
    settings = laid_out(tmp_path)
    run_programs(settings)
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(settings))
    notebook = json.loads((Path(__file__).parents[1] / 'examples' / 'programs.ipynb').read_text())
    code = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
    assert not any(cell.get('outputs') for cell in notebook['cells'])
    namespace = {}
    for source in code:
        exec(source.replace("'my_screen.yaml'", repr(str(config))), namespace)
    plt.close('all')


def test_modules_refuse_loadings_fitted_on_another_matrix(tmp_path):
    settings = laid_out(tmp_path)
    run_programs(settings, steps=['matrix', 'fit'])
    settings['programs']['qval_threshold'] = 1
    run_programs(settings, steps=['matrix'])
    with pytest.raises(ValueError, match='rerun the fit step'):
        run_programs(settings, steps=['modules'])


def test_modules_refuse_scores_of_other_genes(tmp_path):
    settings = laid_out(tmp_path)
    run_programs(settings, steps=['matrix', 'fit'])
    path = tmp_path / 'programs' / 'matrix.parquet'
    pd.read_parquet(path).drop(index='A0').to_parquet(path)
    with pytest.raises(ValueError, match='rerun the fit step'):
        run_programs(settings, steps=['modules'])
