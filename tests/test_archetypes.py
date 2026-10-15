"""
The archetypes stage: path resolution and the commands it builds.

The steps themselves are R scripts, so what is worth testing on this side
is that the right script is called, in the right directory, with the
arguments it reads.  ``dry_run`` reports the commands without running
them, so none of this needs R installed.  The effects step is CHART's
own, so it is tested on a small table of weights instead.
"""

import re
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from scipy.stats import wasserstein_distance

import chart.archetypes
from chart.archetypes import (ENV_DIR, ENV_REPORTS, R_PACKAGES, SCRIPTS,
                              STEPS, run_archetypes, scripts_dir)
from chart.archetypes import effects, workflow
from chart.io.config import EffectsConfig, load_archetypes_config

CHANNEL = 'TOM20'
PARAMS = EffectsConfig(n_perm=20, n_jobs=1)


def config(**overrides):
    section = {'outputs_dir': 'archetype_outputs/',
               'channels': [CHANNEL, 'Golgin97'],
               'params': [[10, 10], [30, 5]]}
    section.update(overrides)
    return {'data_output_dir': '/data/screen',
            'local_output_dir': '/home/me/reports', 'archetypes': section}


def commands(steps=('all',), **overrides):
    planned = run_archetypes(config(**overrides), channels=[CHANNEL],
                             steps=steps, dry_run=True)
    return [' '.join(result.command) for result in planned]


def test_a_channel_works_in_one_directory():
    resolved = load_archetypes_config(config())
    assert (resolved.channel_path(CHANNEL)
            == f'/data/screen/archetype_outputs/{CHANNEL}')


def test_the_directory_is_what_the_scripts_are_told():
    planned = run_archetypes(config(), channels=[CHANNEL], steps=['evaluate'],
                             dry_run=True)
    assert planned[0].directory == f'/data/screen/archetype_outputs/{CHANNEL}'
    assert ENV_DIR == 'CHART_ARCHETYPE_DIR'


def test_the_space_is_built_before_anything_reads_it():
    assert STEPS[0] == 'pca'


def test_the_pca_step_needs_no_r_script():
    planned = run_archetypes(config(), channels=[CHANNEL], steps=['pca'],
                             scripts='/nowhere', dry_run=True)
    assert len(planned) == 1
    assert 'Rscript' not in planned[0].command


def test_the_pca_step_reads_what_the_guide_stage_wrote():
    resolved = load_archetypes_config(config())
    assert (resolved.features_path == '/data/screen/guide_filtering/'
                                      'guide_filtered/allwells-features_cell.parquet')


def test_a_run_without_every_channel_named_is_refused():
    with pytest.raises(ValueError, match='all_channels'):
        run_archetypes(config(), channels=[CHANNEL], steps=['pca'])


def test_diagnostics_go_to_the_reports_root():
    resolved = load_archetypes_config(config())
    assert resolved.report_path(CHANNEL) == f'/home/me/reports/archetypes/{CHANNEL}'


def test_the_reports_directory_is_what_the_scripts_are_told():
    planned = run_archetypes(config(), channels=[CHANNEL], steps=['selectk'],
                             dry_run=True)
    assert planned[0].reports == f'/home/me/reports/archetypes/{CHANNEL}'
    assert ENV_REPORTS == 'CHART_ARCHETYPE_REPORT_DIR'


def test_every_script_writing_diagnostics_reads_the_reports_variable():
    # Python and R agree on this only by the name matching, so a rename on
    # one side has to fail here rather than quietly write to the data root.
    writers = ['pick_npc.R', 'metacell_adaptive_npc.R',
               'eval_spaces_perchannel.R', 'robustAA/test_evalK_onfullPC.R']
    for name in writers:
        assert ENV_REPORTS in (scripts_dir() / name).read_text(), name


def test_a_pair_written_as_one_string_is_read_as_two_numbers():
    resolved = load_archetypes_config(config(params=['10 10', [30, 5]]))
    assert resolved.params == [['10', '10'], [30, 5]]


def test_a_pair_that_is_not_a_pair_is_reported():
    with pytest.raises(ValueError, match='archetypes.params'):
        load_archetypes_config(config(params=[[10, 10, 10]]))


def test_a_pair_that_is_not_numbers_is_reported():
    with pytest.raises(ValueError, match='params'):
        load_archetypes_config(config(params=[['ten', 10]]))


def test_every_step_runs_in_order():
    assert [line.split('.R')[0].rsplit('/', 1)[-1] for line in commands()] == [
        'pca [in CHART] TOM20',
        'pick_npc',
        'metacell_adaptive_npc', 'metacell_adaptive_npc',
        'eval_spaces_perchannel',
        'robust_archetyping', 'robust_archetyping',
        'test_evalK_onfullPC',
        'archetype_weights',
        'effects [in CHART] TOM20',
    ]


def test_the_weights_come_after_the_k_they_need():
    assert STEPS.index('weights') > STEPS.index('selectk')


def test_a_configured_k_reaches_the_steps_that_use_it():
    for step in ('selectk', 'weights'):
        line = commands(steps=[step], k={CHANNEL: 12})[0]
        assert line.endswith(f'{CHANNEL} FALSE 10 10 12'), step


def test_without_a_configured_k_the_steps_find_it_themselves():
    # selectk picks it and writes k.txt; weights reads that.  Neither can
    # be told on the command line here, because the whole run is planned
    # before selectk has written anything.
    for step in ('selectk', 'weights'):
        line = commands(steps=[step])[0]
        assert line.endswith(f'{CHANNEL} FALSE 10 10'), step


def test_the_weights_step_works_from_the_chosen_pair():
    line = commands(steps=['weights'], selected_params=[30, 5])[0]
    assert line.endswith(f'{CHANNEL} FALSE 30 5')


def test_the_steps_that_vary_run_once_per_pair():
    assert len(commands(steps=['metacells'])) == 2
    assert len(commands(steps=['evaluate'])) == 1


def test_a_configured_npc_is_passed_to_the_scripts():
    with_npc = commands(steps=['npc', 'metacells'], npc={CHANNEL: 14})
    assert with_npc[0].endswith('npc.txt 14')
    assert with_npc[1].endswith(f'10 10 {CHANNEL} 14')


def test_without_a_configured_npc_the_scripts_choose_it():
    without = commands(steps=['npc', 'metacells'])
    assert without[0].endswith('npc.txt')
    assert without[1].endswith(f'10 10 {CHANNEL}')


def test_a_step_needing_pairs_says_so_when_there_are_none():
    with pytest.raises(ValueError, match='archetypes.params'):
        run_archetypes(config(params=[]), channels=[CHANNEL],
                       steps=['metacells'], dry_run=True)


def test_no_channels_at_all_is_reported():
    with pytest.raises(ValueError, match='--channel'):
        run_archetypes(config(channels=[]), dry_run=True)


def test_a_missing_script_names_the_way_to_point_at_them():
    with pytest.raises(FileNotFoundError, match='--scripts-dir'):
        run_archetypes(config(), channels=[CHANNEL], scripts='/nowhere',
                       dry_run=True)


def test_the_scripts_ship_inside_the_package():
    # Inside, so that an install carries them.  A path that climbed out of
    # the package would pass in a checkout and fail once installed.
    assert scripts_dir().is_relative_to(Path(chart.archetypes.__file__).parent)
    assert {script.name for script in scripts_dir().glob('*.R')} >= {
        'pick_npc.R', 'metacell_adaptive_npc.R', 'eval_spaces_perchannel.R'}
    # One per R script, plus the pca and effects steps CHART runs itself.
    assert len(STEPS) == 8


def test_every_script_a_step_names_is_there():
    # The packaging names the scripts by glob, so a script in a directory
    # no glob reaches would be missing from an install only.
    for step, script in SCRIPTS.items():
        assert (scripts_dir() / script).exists(), step


def test_the_r_packages_checked_for_are_the_ones_the_scripts_load():
    # The list is written out rather than read off the scripts, so this
    # is what stops the two drifting.  dplyr and ggplot2 were missing
    # from it once, and ran only because SuperCell happens to pull them.
    loaded = set()
    for script in scripts_dir().rglob('*.R'):
        loaded |= set(re.findall(r'(?:library|require)\(([A-Za-z0-9._]+)\)',
                                 script.read_text()))
    assert loaded == set(R_PACKAGES)


def test_a_missing_r_is_reported_before_any_step_runs():
    with mock.patch.object(workflow.shutil, 'which', return_value=None):
        with pytest.raises(RuntimeError, match='environment.yml'):
            run_archetypes(config(), channels=[CHANNEL], steps=['metacells'])


def test_a_dry_run_asks_nothing_of_r():
    # Planning is what --dry-run is for, so it has to work on a machine
    # with no R at all.
    with mock.patch.object(workflow, '_check_r') as checked:
        commands(steps=['all'])
    checked.assert_not_called()


def test_the_pca_step_alone_asks_nothing_of_r():
    with mock.patch.object(workflow, '_check_r') as checked:
        with mock.patch.object(workflow, '_run_pca'):
            run_archetypes(config(), channels=[CHANNEL], steps=['pca'])
    checked.assert_not_called()


def test_the_effects_step_alone_asks_nothing_of_r():
    with mock.patch.object(workflow, '_check_r') as checked:
        with mock.patch.object(workflow, '_run_effects') as ran:
            run_archetypes(config(), channels=[CHANNEL], steps=['effects'])
    checked.assert_not_called()
    ran.assert_called_once()


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
    assert (effects.wasserstein_1d_same_support(v, p, q)
            == pytest.approx(wasserstein_distance(v, v, p, q)))


def test_each_archetype_is_seeded_by_its_gene_and_position():
    table, _ = effects.gene_effects(weights(), 'CALR', params=PARAMS)
    data = weights()
    use = data.Gene.isin(['GENEB', 'NTC', 'OR1A1']).to_numpy()
    expected = effects.w1_with_stratified_permutation(
        effects.clr_transform(data.filter(like='Archetype').to_numpy())[use, 2],
        (data.Gene[use] == 'GENEB').to_numpy(), data.Well[use].to_numpy(),
        PARAMS.n_perm, PARAMS.seed + 1 + 2, PARAMS.trimmed_alpha)
    row = table[(table.Gene == 'GENEB') & (table.Archetype == 'CALR_Archetype3')]
    assert row.pval.item() == expected['pval']
    assert row.w1.item() == expected['w1']


def test_leaving_an_archetype_out_changes_nothing_else():
    full, skipped = effects.gene_effects(weights(), 'CALR', params=PARAMS)
    fewer, _ = effects.gene_effects(weights(), 'CALR', ['CALR_Archetype2'],
                                    PARAMS)
    pd.testing.assert_frame_equal(
        fewer, full[full.Archetype != 'CALR_Archetype2'].reset_index(drop=True))
    assert skipped == ['GENEC']


def test_an_archetype_the_channel_lacks_cannot_be_removed():
    with pytest.raises(ValueError, match='CALR_Archetype9'):
        effects.gene_effects(weights(), 'CALR', ['CALR_Archetype9'], PARAMS)


def with_npc_txt(tmp_path, **overrides):
    resolved = load_archetypes_config({**config(k={CHANNEL: 4}, **overrides),
                                       'data_output_dir': str(tmp_path)})
    directory = Path(resolved.channel_path(CHANNEL))
    directory.mkdir(parents=True)
    (directory / 'npc.txt').write_text('8\n')
    return resolved


def test_missing_weights_are_reported_by_path(tmp_path):
    with pytest.raises(FileNotFoundError,
                       match=f'robust_archweights-{CHANNEL}_k4_gamma10_knn10_npc8'):
        with_npc_txt(tmp_path).weights_path(CHANNEL)


def test_a_configured_npc_must_agree_with_npc_txt(tmp_path):
    with pytest.raises(ValueError, match='npc.txt holds 8'):
        with_npc_txt(tmp_path, npc={CHANNEL: 12}).weights_path(CHANNEL)
