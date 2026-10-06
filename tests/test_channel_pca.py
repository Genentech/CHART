"""
Choosing one channel's features, and the space built from them.

Column names follow the real ones: <compartment>_<measure>_<channel>_<n>
for a single channel, two channels for a colocalisation measure, and no
channel at all for a geometry measure.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chart.dimred.channels import (INDEX_COLUMNS, NoFeaturesError,
                                   channel_features, write_channel_pca)
from chart.io.tables import table_columns

CHANNELS = ['TOM20', 'DAPI1', 'CALR']

COLUMNS = [
    'nuclei_mean-frac_TOM20_0', 'cytosol_pftas_TOM20_7', 'cell_radial-cv_TOM20_2',
    'nuclei_mean-frac_DAPI1_0', 'cytosol_pftas_CALR_3',
    'nuclei_manders_TOM20_DAPI1', 'nuclei_rwc_DAPI1_TOM20',
    'nuclei_corr_TOM20_CALR',
    'nuclei_eccentricity', 'cell_axismajorlength', 'nuclei_humoments_1',
] + list(INDEX_COLUMNS)


def test_a_channels_own_measures_are_kept():
    kept = channel_features(COLUMNS, 'TOM20', CHANNELS)
    assert kept == ['nuclei_mean-frac_TOM20_0', 'cytosol_pftas_TOM20_7',
                    'cell_radial-cv_TOM20_2']


def test_measures_between_two_channels_are_dropped():
    kept = channel_features(COLUMNS, 'TOM20', CHANNELS)
    assert not [name for name in kept if 'manders' in name or 'rwc' in name
                or 'corr' in name]


def test_shape_measures_are_dropped():
    kept = channel_features(COLUMNS, 'TOM20', CHANNELS)
    assert not [name for name in kept
                if 'eccentricity' in name or 'axismajorlength' in name
                or 'humoments' in name]


def test_what_identifies_a_cell_is_not_a_feature():
    kept = channel_features(COLUMNS, 'TOM20', CHANNELS)
    assert not set(kept) & set(INDEX_COLUMNS)


def test_a_channel_missing_from_the_list_is_reported():
    # The dangerous case: removing nothing would keep every other stain.
    with pytest.raises(NoFeaturesError, match='removing the other'):
        channel_features(COLUMNS, 'TOM20', ['DAPI1', 'CALR'])


def test_a_channel_with_nothing_of_its_own_is_reported():
    with pytest.raises(NoFeaturesError, match='belong to TOM20 alone'):
        channel_features(['nuclei_manders_TOM20_DAPI1', 'nuclei_eccentricity'],
                         'TOM20', CHANNELS)


def screen(path, cells=60):
    """A feature table shaped like the real one."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {name: rng.normal(size=cells)
         for name in COLUMNS if name not in INDEX_COLUMNS})
    frame['Well'] = 'A1'
    frame['Label'] = np.arange(cells)
    frame['Guide'] = [f'g{i % 6}' for i in range(cells)]
    frame['Gene'] = ['NTC' if i % 3 == 0 else f'GENE{i % 4}'
                     for i in range(cells)]
    frame.to_parquet(path)
    return frame


def test_the_space_is_written_where_the_scripts_look(tmp_path):
    features = str(tmp_path / 'allwells-features_cell.parquet')
    screen(features)

    written = write_channel_pca(features, 'TOM20', CHANNELS,
                                str(tmp_path / 'TOM20'))
    assert written.endswith('TOM20/cell_pca_centered.parquet')


def test_the_written_columns_are_what_load_pca_expects(tmp_path):
    features = str(tmp_path / 'allwells-features_cell.parquet')
    screen(features)
    written = write_channel_pca(features, 'TOM20', CHANNELS,
                                str(tmp_path / 'TOM20'))

    columns = table_columns(written)
    # load_pca() drops the last four and reads the rest as components.
    assert tuple(columns[-4:]) == INDEX_COLUMNS
    assert all(name.startswith('PC') for name in columns[:-4])
    assert len(columns) > 4


def test_every_cell_survives_the_step(tmp_path):
    features = str(tmp_path / 'allwells-features_cell.parquet')
    frame = screen(features)
    written = write_channel_pca(features, 'TOM20', CHANNELS,
                                str(tmp_path / 'TOM20'))
    assert len(pd.read_parquet(written)) == len(frame)


def test_the_space_is_centred_on_the_controls(tmp_path):
    features = str(tmp_path / 'allwells-features_cell.parquet')
    screen(features)
    written = write_channel_pca(features, 'TOM20', CHANNELS,
                                str(tmp_path / 'TOM20'))

    table = pd.read_parquet(written)
    controls = table[table['Gene'] == 'NTC']
    components = [name for name in table.columns if name.startswith('PC')]
    assert np.allclose(controls[components].mean(axis=0), 0, atol=1e-8)


def test_a_table_that_cannot_name_its_cells_is_reported(tmp_path):
    features = str(tmp_path / 'nameless.parquet')
    pd.DataFrame({'nuclei_pftas_TOM20_0': [1.0, 2.0, 3.0]}).to_parquet(features)

    with pytest.raises(ValueError, match='cannot be named'):
        write_channel_pca(features, 'TOM20', CHANNELS, str(tmp_path / 'TOM20'))
