"""
What CHART does when its input is wrong.

Each test breaks one thing about an otherwise healthy well and asserts
that CHART says so.  The cases here all used to run to completion and
report success while writing a table with no rows in it, which is the
failure worth guarding against: an empty result that looks like a
finished one.
"""

import numpy as np
import pandas as pd
import pytest

from chart.filtering.features import NoFeaturesError
from chart.filtering.labels import NoCellsError
from chart.filtering.workflow import run_filtering

WELL = 'A1'
CELLS = 40


class Screen:
    """One well of input, and the knobs to spoil it."""

    def __init__(self, root):
        self.root = root
        self.merged = root / 'preprocessing' / 'bywell'
        self.filters = self.merged / 'filters'

    @property
    def config(self):
        return {
            'data_output_dir': str(self.root),
            'local_output_dir': str(self.root),
            'preprocessing_bywell': {
                'merged_dir': 'preprocessing/bywell/',
                'filter_dir': 'preprocessing/bywell/filters/',
                'filtered_output_dir': 'preprocessing/bywell/filtered/',
            },
        }

    def table(self, suffix):
        return pd.read_parquet(self.merged / f'{WELL}-{suffix}.parquet')

    def write(self, suffix, frame):
        frame.to_parquet(self.merged / f'{WELL}-{suffix}.parquet')

    def write_filters(self, labels):
        pd.DataFrame({'label': labels}).to_parquet(
            self.filters / f'{WELL}_phenocorfilt.parquet')

    def run(self, config=None):
        return run_filtering(config or self.config, wells=[WELL])


@pytest.fixture
def screen(tmp_path):
    """A well that filters cleanly, to break one piece at a time."""
    screen = Screen(tmp_path)
    screen.filters.mkdir(parents=True)

    rng = np.random.default_rng(0)
    identity = pd.DataFrame({
        'label': np.arange(CELLS),
        'sgRNA': [f'g{i % 4}' for i in range(CELLS)],
        'gene_symbol': ['NTC' if i % 4 == 0 else 'OR5A1'
                        for i in range(CELLS)],
    })
    screen.write('objects', identity.assign(
        **{'nuclei_centroid-0': rng.normal(500, 50, CELLS)}))
    screen.write('features', identity.assign(
        cell_area=rng.normal(100, 10, CELLS),
        cell_perimeter=rng.normal(40, 4, CELLS)))
    # A legacy filter name, so no manifest is needed.
    screen.write_filters(np.arange(CELLS))
    return screen


def test_healthy_well_keeps_every_cell(screen):
    result, = screen.run()
    assert (result.objects_kept, result.features_kept) == (CELLS, CELLS)
    assert result.merged_written


def test_label_type_differing_between_tables_is_reported(screen):
    features = screen.table('features')
    screen.write('features', features.assign(
        label=features['label'].astype(str)))

    with pytest.raises(NoCellsError, match='filtered out'):
        screen.run()


def test_labels_identifying_other_objects_are_reported(screen):
    features = screen.table('features')
    screen.write('features', features.assign(label=features['label'] + 1000))

    with pytest.raises(NoCellsError, match='filtered out'):
        screen.run()


def test_input_table_without_rows_is_reported(screen):
    screen.write('features', screen.table('features').iloc[:0])

    with pytest.raises(NoCellsError, match='no rows'):
        screen.run()


def test_filters_naming_no_known_label_are_reported(screen):
    screen.write_filters(np.arange(CELLS) + 1000)

    with pytest.raises(NoCellsError, match='filtered out'):
        screen.run()


def test_objects_missing_from_the_feature_table_are_counted(screen, caplog):
    screen.write('features', screen.table('features').iloc[:-4])

    with caplog.at_level('WARNING'):
        result, = screen.run()

    assert result.merged_written
    assert '4 of the 40 objects' in caplog.text


def test_missing_input_file_is_reported(screen):
    (screen.merged / f'{WELL}-features.parquet').unlink()

    with pytest.raises(FileNotFoundError, match='features'):
        screen.run()


def test_feature_patterns_matching_nothing_are_reported(screen):
    config = screen.config
    config['schema'] = {'feature_patterns': ['Cells_']}

    with pytest.raises(NoFeaturesError, match='Cells_'):
        screen.run(config)
