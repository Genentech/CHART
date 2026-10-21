"""
The archetypes stage: path resolution and the commands it builds.

The steps themselves are R scripts, so what is worth testing on this side
is that the right script is called, in the right directory, with the
arguments it reads.  ``dry_run`` reports the commands without running
them, so none of this needs R installed.  The crops step is CHART's own,
and runs here on small made-up images in place of what scallops reads.
"""

import re
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import chart.archetypes
from chart.archetypes import (ENV_DIR, ENV_REPORTS, R_PACKAGES, SCRIPTS,
                              STEPS, run_archetypes, scripts_dir)
from chart.archetypes import crops, workflow
from chart.io.config import load_archetypes_config

CHANNEL = 'TOM20'


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
        'crops [in CHART] TOM20',
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


def test_the_archetypes_step_is_told_the_published_sample_sizes():
    line = commands(steps=['archetypes'])[0]
    assert line.endswith(f'10 10 {CHANNEL} FALSE 12000 8000 12000')


def test_configured_sample_sizes_reach_the_archetypes_step():
    line = commands(steps=['archetypes'], coverage_sample_size=600,
                    eval_sample_size=400, fit_sample_size=500)[0]
    assert line.endswith(f'10 10 {CHANNEL} FALSE 600 400 500')


def test_a_sample_size_that_is_not_a_count_is_reported():
    for bad in (0, -5, 1.5, 'many', True):
        with pytest.raises(ValueError, match='fit_sample_size'):
            load_archetypes_config(config(fit_sample_size=bad))


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
    # One per R script, plus the pca and crops steps CHART runs itself.
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


def test_the_crops_come_after_the_weights_they_rank():
    assert STEPS[-1] == 'crops'


def test_the_crops_step_alone_asks_nothing_of_r():
    with mock.patch.object(workflow, '_check_r') as checked:
        with mock.patch.object(workflow, '_run_crops'):
            run_archetypes(config(), channels=[CHANNEL], steps=['crops'])
    checked.assert_not_called()


def test_the_published_number_of_cells_is_the_default():
    assert load_archetypes_config(config()).top_cells == 10


def test_crop_settings_of_the_wrong_shape_are_reported():
    for settings, message in (({'image_channels': {CHANNEL: -1}}, 'image_channels'),
                              ({'image_channels': {CHANNEL: 'eight'}}, 'image_channels'),
                              ({'top_cells': 0}, 'top_cells'),
                              ({'display_percentiles': {CHANNEL: [99, 1]}},
                               f'display_percentiles.{CHANNEL}'),
                              ({'display_percentiles': {CHANNEL: [0, 101]}},
                               f'display_percentiles.{CHANNEL}'),
                              ({'display_ranges': {CHANNEL: [105, 4000]}},
                               'keyed by well'),
                              ({'display_ranges': {CHANNEL: {'A1': [4000]}}},
                               f'display_ranges.{CHANNEL}.A1')):
        with pytest.raises(ValueError, match=re.escape(message)):
            load_archetypes_config(config(**settings))


class FakeImage:
    """Enough of what scallops returns for the crops step: a channel picked
    by position, then the single dimensions dropped."""

    def __init__(self, array):
        self.array = array

    def isel(self, c):
        return FakeImage(self.array[:, c])

    def squeeze(self):
        return self.array.squeeze()


def crop_config(tmp_path, monkeypatch, archetypes=None, wells=None):
    """Twelve cells in one well, with two archetypes' weights, their boxes
    in the objects table and the images they are cropped from.

    Each cell is a ten-pixel square on a grid whose first square is in
    the image's corner.  The first archetype weights the cells by label,
    the second the other way round, so its top cell is the corner one.
    """
    data, channel_dir = tmp_path / 'data', tmp_path / 'data' / 'archetype_outputs' / CHANNEL
    channel_dir.mkdir(parents=True)
    (channel_dir / 'npc.txt').write_text('8\n')
    labels = np.arange(1, 13)
    pd.DataFrame({'Archetype1': labels / 12, 'Archetype2': 1 - labels / 12,
                  'Well': 'A1', 'Label': labels, 'Guide': 'g1', 'Gene': 'NTC'}
                 ).to_parquet(channel_dir / f'robust_archweights-{CHANNEL}'
                                            f'_k2_gamma30_knn10_npc8.parquet')

    y0, x0 = (labels - 1) // 4 * 20, (labels - 1) % 4 * 20
    combined = data / 'preprocessing' / 'allwells' / 'combined'
    combined.mkdir(parents=True)
    pd.DataFrame({'cell_bbox-0': y0, 'cell_bbox-1': x0,
                  'cell_bbox-2': y0 + 10, 'cell_bbox-3': x0 + 10},
                 index=pd.MultiIndex.from_arrays(
                     [labels, ['g1'] * 12, ['NTC'] * 12, ['A1'] * 12],
                     names=['Label', 'Guide', 'Gene', 'Well'])
                 ).to_parquet(combined / 'allwells-objects.parquet')

    cells, nuclei = np.zeros((2, 100, 100), dtype=np.int32)
    for label, y, x in zip(labels, y0, x0):
        cells[y:y + 10, x:x + 10] = label
        nuclei[y + 3:y + 7, x + 3:x + 7] = label
    stain = np.random.default_rng(0).integers(0, 4000, size=(1, 3, 1, 100, 100))
    experiments = {
        '/images/pheno.zarr': SimpleNamespace(images={'A1': FakeImage(stain)},
                                              labels={}),
        '/images/cells.zarr': SimpleNamespace(
            images={}, labels={'A1-cell': FakeImage(cells[None, None, None])}),
        '/images/nuclei.zarr': SimpleNamespace(
            images={}, labels={'A1-nuclei': FakeImage(nuclei[None, None, None])})}
    monkeypatch.setattr(crops, 'read_experiment',
                        lambda path, dask, group_by: experiments[path])

    section = {'outputs_dir': 'archetype_outputs/', 'channels': [CHANNEL],
               'params': [[30, 10]], 'k': {CHANNEL: 2},
               'image_channels': {CHANNEL: 1}, 'top_cells': 7}
    section.update(archetypes or {})
    images = {'pheno_dir': 'pheno.zarr', 'cell_labels_dir': 'cells.zarr',
              'nuclei_labels_dir': 'nuclei.zarr'}
    images.update(wells or {})
    return {'data_output_dir': str(data),
            'local_output_dir': str(tmp_path / 'reports'),
            'preprocessing_bywell': {'scallops_dir': '/images',
                                     'global': images},
            'archetypes': section}


def crops_dir(tmp_path):
    return tmp_path / 'reports' / 'archetypes' / CHANNEL / crops.CROPS_DIR


def test_each_archetype_shows_its_most_heavily_weighted_cells(tmp_path, monkeypatch):
    run_archetypes(crop_config(tmp_path, monkeypatch), channels=[CHANNEL],
                   steps=['crops'])
    shown = pd.read_parquet(crops_dir(tmp_path) / 'top_cells.parquet')
    first = shown[shown['Archetype'] == 'Archetype1']
    assert first['Label'].tolist() == [12, 11, 10, 9, 8, 7, 6]
    assert first['Rank'].tolist() == list(range(1, 8))
    second = shown[shown['Archetype'] == 'Archetype2']
    assert second['Label'].tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert second.iloc[0][['cell_bbox-0', 'cell_bbox-1', 'cell_bbox-2',
                           'cell_bbox-3']].tolist() == [0, 0, 10, 10]


def test_the_grids_are_named_as_the_annotation_reads_them(tmp_path, monkeypatch):
    # Seven cells make a full row of five and a second of two.
    run_archetypes(crop_config(tmp_path, monkeypatch), channels=[CHANNEL],
                   steps=['crops'])
    assert sorted(p.name for p in crops_dir(tmp_path).glob('*.png')) == [
        f'archetype{n}_{CHANNEL}_grid_{row}.png' for n in (1, 2) for row in (1, 2)]


def test_a_cell_at_the_edge_of_the_image_is_cropped_inside_it():
    assert crops.crop_bounds((0, 0, 10, 10), (100, 100)) == (0, 0, 18, 18)
    assert crops.crop_bounds((90, 95, 100, 100), (100, 100)) == (82, 87, 100, 100)
    assert crops.crop_bounds((40, 40, 50, 50), (100, 100)) == (32, 32, 58, 58)


def test_the_display_range_is_taken_from_patches_inside_the_margin(monkeypatch):
    # Patches of ten pixels at rows and columns 30, 50 and 70, so the
    # lowest pixel is the upper-left patch's first and the highest the
    # lower-right patch's last.
    monkeypatch.setattr(crops, 'PATCH_SIZE', 10)
    image = np.arange(10000).reshape(100, 100)
    assert crops.display_range(image, (0, 100)) == (3030.0, 7979.0)


def test_a_configured_display_range_is_used_for_its_well(tmp_path, monkeypatch, caplog):
    caplog.set_level('INFO')
    run_archetypes(crop_config(tmp_path, monkeypatch,
                               {'display_ranges': {CHANNEL: {'A1': [105, 4000]}}}),
                   channels=[CHANNEL], steps=['crops'])
    assert 'shown from 105 to 4000, as configured' in caplog.text


def test_a_channel_without_an_image_channel_is_skipped_unless_asked_for(
        tmp_path, monkeypatch, caplog):
    source = crop_config(tmp_path, monkeypatch, {'image_channels': {}})
    resolved = load_archetypes_config(source)
    assert crops.write_channel_crops(source, resolved, CHANNEL) == []
    assert 'Skipping the crops' in caplog.text
    with pytest.raises(crops.CropsNotSetUpError, match='image_channels'):
        run_archetypes(source, channels=[CHANNEL], steps=['crops'])


def test_without_scallops_the_crops_are_skipped(tmp_path, monkeypatch, caplog):
    source = crop_config(tmp_path, monkeypatch)
    monkeypatch.setattr(crops, 'read_experiment', None)
    assert crops.write_channel_crops(source, load_archetypes_config(source),
                                     CHANNEL) == []
    assert 'scallops' in caplog.text


def test_a_well_without_its_label_images_is_reported(tmp_path, monkeypatch):
    source = crop_config(tmp_path, monkeypatch, wells={'nuclei_labels_dir': ''})
    with pytest.raises(ValueError, match="'nuclei_labels_dir'.*A1"):
        run_archetypes(source, channels=[CHANNEL], steps=['crops'])


def test_a_cell_the_objects_table_has_no_box_for_is_reported(tmp_path, monkeypatch):
    source = crop_config(tmp_path, monkeypatch)
    path = tmp_path / 'data' / 'preprocessing' / 'allwells' / 'combined' / 'allwells-objects.parquet'
    pd.read_parquet(path).iloc[1:].to_parquet(path)
    with pytest.raises(ValueError, match='no bounding box.*cell 1 of well A1'):
        run_archetypes(source, channels=[CHANNEL], steps=['crops'])


def test_cells_with_no_outline_where_their_box_is_are_warned_about(
        tmp_path, monkeypatch, caplog):
    source = crop_config(tmp_path, monkeypatch)
    blank = SimpleNamespace(images={}, labels={
        'A1-cell': FakeImage(np.zeros((1, 1, 1, 100, 100), dtype=np.int32))})
    read = crops.read_experiment
    monkeypatch.setattr(crops, 'read_experiment', lambda path, dask, group_by: (
        blank if path == '/images/cells.zarr' else read(path, dask, group_by)))
    run_archetypes(source, channels=[CHANNEL], steps=['crops'])
    assert '14 of the 14 cells shown' in caplog.text
