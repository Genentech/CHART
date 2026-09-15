"""
Configuration loading and path resolution.

This module contains functions for loading configuration files and
resolving the output and well-specific paths derived from them.

:func:`load_bywell_config` turns a raw configuration mapping into
a :class:`BywellConfig`, i.e. the list of settings CHART understands,
together with their defaults.
"""

import json
import logging
import re
import yaml
from dataclasses import dataclass, field, fields, replace
from typing import Any, Dict, List, Optional, Union

from .tables import list_dir

logger = logging.getLogger(__name__)


def load_config_from_file(config_file: str) -> Dict[str, Any]:
    """
    Load configuration parameters from a JSON or YAML file.
    
    Args:
        config_file: Path to configuration file (.json or .yaml/.yml)
    
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_file, 'r') as f:
        if config_file.lower().endswith(('.yaml', '.yml')):
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    return config


def join_path(base_dir: str, path: str) -> str:
    """
    Join base directory with a path, handling both absolute and relative paths.
    
    Args:
        base_dir: Base directory (scallops_dir, data_output_dir, or local_output_dir)
        path: Path to join (can be absolute or relative)
    
    Returns:
        Joined path
    """
    if not base_dir:
        return path
    
    # If path is already absolute (starts with s3://, http://, /, etc.), return as is
    if path.startswith(('s3://', 'http://', 'https://', '/', 'C:', 'D:')):
        return path
    
    # Join base_dir with relative path
    return base_dir.rstrip('/') + '/' + path.lstrip('/')


def get_data_output_path(config: Dict[str, Any], subpath: str = "") -> str:
    """
    Get the full path for data outputs (large files like parquet, zarr).
    
    Args:
        config: Configuration dictionary
        subpath: Subpath to append to data output directory
    
    Returns:
        Full path for data outputs
    """
    data_output_dir = config.get('data_output_dir', '')
    return join_path(data_output_dir, subpath)


def get_local_output_path(config: Dict[str, Any], subpath: str = "") -> str:
    """
    Get the full path for local outputs (plots, reports, small files).
    
    Args:
        config: Configuration dictionary
        subpath: Subpath to append to local output directory
    
    Returns:
        Full path for local outputs
    """
    local_output_dir = config.get('local_output_dir', '')
    return join_path(local_output_dir, subpath)


def get_well_config(config: Dict[str, Any], well: str) -> Dict[str, Any]:
    """
    Get well-specific configuration by merging global parameters with well overrides.
    
    Args:
        config: Full configuration dictionary
        well: Well identifier
    
    Returns:
        Merged configuration for the specific well
    """
    # Start with global parameters
    well_config = config.get('global', {}).copy()
    
    # Apply well-specific overrides
    if 'wells' in config and well in config['wells']:
        well_overrides = config['wells'][well]
        # Handle case where well_overrides might be None or empty
        if well_overrides:
            well_config.update(well_overrides)
    
    return well_config


SECTION_NAME = 'preprocessing_bywell'

# Keys that belong to other stages of the pipeline.  They are listed so the
# loader can stay quiet about them: warning on every key CHART does not read
# would bury the warnings that matter under a dozen false alarms, since the
# configuration file is shared across stages.
_OTHER_STAGE_TOP_LEVEL = frozenset()
_OTHER_STAGE_SECTION = frozenset({'barcode_file', 'labels', 'feature_types'})
_OTHER_STAGE_WELL = frozenset({'feature_dir', 'reads_file_pattern', 'reads_file',
                               'filter_duplicates', 'barcode_colname'})

ALLWELLS_SECTION = 'preprocessing_allwells'

# Every key of the across-wells section is now read, so there is nothing
# left to hold back from validation.
_LATER_STEP_ALLWELLS = frozenset()
_ALLWELLS_KEYS = frozenset({'combined_dir', 'normalized_dir', 'plot_dir',
                            'normalization_type', 'filter_dir',
                            'outlier_filtered_dir', 'contamination',
                            'random_state', 'n_estimators',
                            'missing_value_threshold', 'rcv_threshold'})

GUIDE_SECTION = 'guide_filtering'

# cellmapp calls the guide step's own output directory 'outlier_filtered_dir',
# copied from the across-wells section where it means the input.  The name is
# kept so one configuration file still serves both codebases; the property
# below gives it an accurate name on our side.
_GUIDE_KEYS = frozenset({'filter_dir', 'plot_dir', 'outlier_filtered_dir',
                         'cosine_similarity_threshold', 'min_cells_per_guide'})

SCHEMA_SECTION = 'schema'

_TOP_LEVEL_KEYS = frozenset({'data_output_dir', 'local_output_dir',
                             SECTION_NAME, ALLWELLS_SECTION, GUIDE_SECTION,
                             SCHEMA_SECTION})
_SECTION_KEYS = frozenset({'scallops_dir', 'merged_dir', 'filter_dir', 'qc_dir',
                           'filtered_output_dir', 'premerged',
                           'objects_pattern', 'features_pattern',
                           'merged_pattern',
                           'column_mapping',
                           'column_prefix_mapping', 'exclude_patterns',
                           'drop_unassigned', 'precomputed_filters',
                           'thresholds', 'global', 'wells'})


#: The index level names CHART uses once a table is in hand, whatever the
#: input called its columns.  Fixed, unlike the schema below: the caller's
#: names are mapped onto these on the way in and restored on the way out.
LABEL_LEVEL = 'Label'
GUIDE_LEVEL = 'Guide'
GENE_LEVEL = 'Gene'
WELL_LEVEL = 'Well'

#: Which index levels each ``level`` keeps, before ``Well`` is appended.
#: Note that nothing is aggregated: the rows are still one per cell.
INDEX_LEVELS = {
    'cell': (LABEL_LEVEL, GUIDE_LEVEL, GENE_LEVEL),
    'guide': (GUIDE_LEVEL, GENE_LEVEL),
    'gene': (GENE_LEVEL,),
}


@dataclass
class SchemaConfig:
    """Where each role lives in the caller's tables, and what the values mean.
    """
    cell_column: str = 'label'
    guide_column: str = 'sgRNA'
    gene_column: str = 'gene_symbol'

    #: Regular expressions marking a column as a feature.  Unanchored,
    #: so 'nuclei_' matches anywhere in the name and '^nuclei_' only at
    #: the start.
    feature_patterns: List[str] = field(
        default_factory=lambda: ['cell_', 'nuclei_', 'cytosol_'])
    #: Columns to carry alongside the features rather than measure.
    metadata: Optional[List[str]] = None

    #: Which gene values are controls: this one, or this prefix.
    control_gene: str = 'NTC'
    control_prefix: str = 'OR'

    #: Where the outlier position plot reads cell coordinates.
    centroid_columns: List[str] = field(
        default_factory=lambda: ['nuclei_centroid-0', 'nuclei_centroid-1'])

    def rename_map(self) -> Dict[str, str]:
        """Map the caller's column names onto the ones the steps read.

        Only the roles that differ are listed, so a default schema
        renames nothing.
        """
        expected = {'cell_column': 'label', 'guide_column': 'sgRNA',
                    'gene_column': 'gene_symbol'}
        return {getattr(self, field_name): name
                for field_name, name in expected.items()
                if getattr(self, field_name) != name}


@dataclass
class Thresholds:
    """Cutoffs applied by the QC components."""
    pheno_correlation: float = 0.8
    pheno_sbs_correlation: float = 0.8
    segmentation_error_threshold: float = 15.0


@dataclass
class WellConfig:
    """Per-well settings, with the ``global`` block supplying the defaults."""
    well: str = ''
    pheno_dir: str = ''
    pheno_sbs_registered_dir: str = ''
    sbs_dir: str = ''
    sbs_feature_dir: str = ''
    dapi_channel_names: Optional[List[str]] = None
    pheno_sbs_dapi_channel: Union[str, int, None] = None

    def resolve_pheno_sbs_dapi_channel(self) -> Union[str, int]:
        """Which pheno channel holds the DAPI stain in SBS-registered space."""
        if self.pheno_sbs_dapi_channel is not None:
            return self.pheno_sbs_dapi_channel
        if self.dapi_channel_names:
            return self.dapi_channel_names[-1]
        raise ValueError(
            f"cannot determine the pheno DAPI channel for well {self.well}: set "
            f"'pheno_sbs_dapi_channel' (a channel name, or an integer index for "
            f"legacy data) or 'dapi_channel_names'"
        )


@dataclass
class BywellConfig:
    """The by-well preprocessing settings, with their defaults in one place."""
    data_output_dir: str
    local_output_dir: str
    scallops_dir: str = ''
    merged_dir: str = 'preprocessing/bywell/'
    filter_dir: str = 'preprocessing/bywell/filters/'
    qc_dir: str = 'preprocessing/bywell/qc_reports/'
    filtered_output_dir: str = 'preprocessing/bywell/filtered/'
    premerged: bool = False

    # Input filenames, when the caller's are not {well}-objects.parquet and
    # {well}-features.parquet.  '{well}' is substituted; the extension
    # decides the reader.
    objects_pattern: Optional[str] = None
    features_pattern: Optional[str] = None
    #: Used instead of the two above when ``premerged`` is set, as there
    #: is then one file holding both.
    merged_pattern: Optional[str] = None
    column_mapping: Optional[Dict[str, str]] = None
    column_prefix_mapping: Optional[Dict[str, str]] = None
    exclude_patterns: Optional[List[str]] = None
    drop_unassigned: bool = False
    precomputed_filters: Optional[Dict[str, Any]] = None
    thresholds: Thresholds = field(default_factory=Thresholds)
    schema: SchemaConfig = field(default_factory=SchemaConfig)
    wells: Dict[str, WellConfig] = field(default_factory=dict)
    defaults: WellConfig = field(default_factory=WellConfig)

    # Directories, resolved against the relevant output root.  Bulk data goes
    # to data_output_dir; reports and filter lists stay local.
    @property
    def merged_path(self) -> str:
        return join_path(self.data_output_dir, self.merged_dir)

    @property
    def filtered_path(self) -> str:
        return join_path(self.data_output_dir, self.filtered_output_dir)

    @property
    def filters_path(self) -> str:
        return join_path(self.local_output_dir, self.filter_dir)

    @property
    def reports_path(self) -> str:
        return join_path(self.local_output_dir, self.qc_dir)

    def well(self, name: str) -> WellConfig:
        """Settings for *name*, falling back to the global block when the
        well is not listed in the configuration."""
        if name in self.wells:
            return self.wells[name]
        return replace(self.defaults, well=name)

    def well_names(self) -> List[str]:
        """Every well to process.

        The ``wells`` section names them when it is present.  It carries
        image directories, which only the QC step reads, so a run that
        stops at normalisation need not list anything; the wells are then
        read off the input filenames instead.
        """
        if self.wells:
            return list(self.wells)
        return self.discover_wells()

    def discover_wells(self) -> List[str]:
        """Which wells the input directory holds tables for.

        Raises:
            ValueError: If nothing matches, since a run over no wells
                would otherwise report success having done nothing.
        """
        pattern = ((self.merged_pattern or '{well}.parquet') if self.premerged
                   else (self.objects_pattern or '{well}-objects.parquet'))
        head, marker, tail = pattern.partition('{well}')
        if not marker:
            raise ValueError(
                f"Cannot tell which wells '{pattern}' covers, as it holds no "
                f"'{{well}}' placeholder; name the wells under "
                f"'{SECTION_NAME}' instead")

        matcher = re.compile(re.escape(head) + '(.+)' + re.escape(tail) + '$')
        found = sorted({match.group(1)
                        for match in (matcher.match(entry)
                                      for entry in list_dir(self.merged_path))
                        if match})
        if not found:
            raise ValueError(
                f"No input tables matching '{pattern}' in {self.merged_path}, "
                f"so there are no wells to process. Check the path and the "
                f"filename pattern, or name the wells under '{SECTION_NAME}'")

        logger.info(f"Found {len(found)} wells in {self.merged_path}: "
                    f"{', '.join(found)}")
        return found


@dataclass
class AllwellsConfig:
    """The across-wells preprocessing settings, with their defaults."""
    data_output_dir: str
    local_output_dir: str
    combined_dir: str = 'preprocessing/allwells/combined/'
    normalized_dir: str = 'preprocessing/allwells/normalized/'
    outlier_filtered_dir: str = 'preprocessing/allwells/outlier_filtered/'
    filter_dir: str = 'preprocessing/allwells/filters/'
    plot_dir: str = 'preprocessing/allwells/plots/'
    normalization_type: str = 'quantile'

    # Outlier detection.  'auto' lets IsolationForest choose how many cells
    # to flag; a float fixes the proportion.
    contamination: Union[str, float] = 'auto'
    random_state: int = 42
    n_estimators: int = 100

    # Column and row prefilters applied before outlier detection.  The missing-value
    # threshold is a count, not a fraction, which is why its default rarely
    # removes anything; see OPEN_ISSUES.md.
    missing_value_threshold: int = 100000
    rcv_threshold: float = 0.01

    schema: SchemaConfig = field(default_factory=SchemaConfig)

    # Bulk data goes to data_output_dir; plots and filter lists stay local,
    # as in the by-well half.
    @property
    def combined_path(self) -> str:
        return join_path(self.data_output_dir, self.combined_dir)

    @property
    def normalized_path(self) -> str:
        return join_path(self.data_output_dir, self.normalized_dir)

    @property
    def outlier_filtered_path(self) -> str:
        return join_path(self.data_output_dir, self.outlier_filtered_dir)

    @property
    def filters_path(self) -> str:
        return join_path(self.local_output_dir, self.filter_dir)

    @property
    def plots_path(self) -> str:
        return join_path(self.local_output_dir, self.plot_dir)


@dataclass
class GuideConfig:
    """The guide filtering settings, with their defaults.

    The step's input is the outlier step's output, so it is read from
    :class:`AllwellsConfig` rather than named here.
    """
    data_output_dir: str
    local_output_dir: str
    filter_dir: str = 'guide_filtering/filters/'
    plot_dir: str = 'guide_filtering/plots/'
    outlier_filtered_dir: str = 'guide_filtering/guide_filtered/'

    cosine_similarity_threshold: float = 0.2
    min_cells_per_guide: int = 50

    schema: SchemaConfig = field(default_factory=SchemaConfig)

    @property
    def guide_filtered_path(self) -> str:
        """Where the guide-filtered tables go, named for what it holds."""
        return join_path(self.data_output_dir, self.outlier_filtered_dir)

    @property
    def filters_path(self) -> str:
        return join_path(self.local_output_dir, self.filter_dir)

    @property
    def plots_path(self) -> str:
        return join_path(self.local_output_dir, self.plot_dir)


@dataclass
class PreprocessingConfig:
    """The three halves of the preprocessing stage.

    The by-well steps, the across-wells steps and the guide step read
    different sections of the same file, so the stage needs all three.
    """
    bywell: BywellConfig
    allwells: AllwellsConfig
    guides: GuideConfig


_THRESHOLD_KEYS = frozenset(f.name for f in fields(Thresholds))
_WELL_KEYS = frozenset(f.name for f in fields(WellConfig)) - {'well'}
_SCHEMA_KEYS = frozenset(f.name for f in fields(SchemaConfig))


def _warn_unknown(values: Dict[str, Any], known, other_stage, where: str) -> None:
    """Report keys that CHART will not read, so typos do not pass silently."""
    for key in values:
        if key not in known and key not in other_stage:
            logger.warning(f"Unknown config key '{key}' in {where}; it will be ignored")


def _as_float(value: Any, key: str, where: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Config key '{key}' in {where} must be a number, "
                         f"got {value!r}") from None


def _build_thresholds(section: Dict[str, Any], config: Dict[str, Any]) -> Thresholds:
    """Read the thresholds, most specific location first."""
    block = section.get('thresholds') or {}
    _warn_unknown(block, _THRESHOLD_KEYS, frozenset(), f'{SECTION_NAME}.thresholds')

    values = {}
    for key in _THRESHOLD_KEYS:
        for source, where in ((block, f'{SECTION_NAME}.thresholds'),
                              (section, SECTION_NAME),
                              (config, 'the top level')):
            if key in source:
                values[key] = _as_float(source[key], key, where)
                break
    return Thresholds(**values)


def _build_schema(config: Dict[str, Any]) -> SchemaConfig:
    """Read the schema, which is shared by every step and so sits at the top."""
    block = config.get(SCHEMA_SECTION) or {}
    if not isinstance(block, dict):
        raise ValueError(f"'{SCHEMA_SECTION}' is present but holds no settings; "
                         f"remove it or fill it in")
    _warn_unknown(block, _SCHEMA_KEYS, frozenset(), SCHEMA_SECTION)
    return SchemaConfig(**{k: v for k, v in block.items() if k in _SCHEMA_KEYS})


def load_bywell_config(source: Union[str, Dict[str, Any], BywellConfig]) -> BywellConfig:
    """Build a :class:`BywellConfig` from a file path or a configuration mapping.
    
    Raises:
        ValueError: if the configuration is not a mapping, if the by-well
            section is present but empty, or if an output directory is
            missing.
    """
    if isinstance(source, BywellConfig):
        return source

    config = load_config_from_file(source) if isinstance(source, str) else source
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping, got {type(config).__name__}")

    if SECTION_NAME in config:
        section = config[SECTION_NAME]
        if not isinstance(section, dict):
            raise ValueError(f"'{SECTION_NAME}' is present but holds no settings; "
                             f"remove it or fill it in")
        # Threshold names are accepted at the top level for backward
        # compatibility, so they are not unknown keys.
        _warn_unknown(config, _TOP_LEVEL_KEYS | _THRESHOLD_KEYS, _OTHER_STAGE_TOP_LEVEL,
                      'the top level')
        _warn_unknown(section, _SECTION_KEYS | _THRESHOLD_KEYS, _OTHER_STAGE_SECTION,
                      SECTION_NAME)
    else:
        # A flat configuration, holding the by-well settings at the top level.
        section = config
        _warn_unknown(config, _TOP_LEVEL_KEYS | _SECTION_KEYS | _THRESHOLD_KEYS,
                      _OTHER_STAGE_TOP_LEVEL | _OTHER_STAGE_SECTION
                      | _ALLWELLS_KEYS | _LATER_STEP_ALLWELLS | _GUIDE_KEYS,
                      'the top level')

    for key in ('data_output_dir', 'local_output_dir'):
        if not config.get(key):
            raise ValueError(
                f"Configuration must set '{key}'. Without it output paths would be "
                f"resolved relative to the working directory"
            )

    _warn_unknown(section.get('global') or {}, _WELL_KEYS, _OTHER_STAGE_WELL,
                  f'{SECTION_NAME}.global')

    wells = {}
    for name, overrides in (section.get('wells') or {}).items():
        _warn_unknown(overrides or {}, _WELL_KEYS, _OTHER_STAGE_WELL,
                      f'{SECTION_NAME}.wells.{name}')
        merged = get_well_config(section, name)
        wells[name] = WellConfig(well=name,
                                 **{k: v for k, v in merged.items() if k in _WELL_KEYS})

    global_values = section.get('global') or {}
    defaults = WellConfig(**{k: v for k, v in global_values.items() if k in _WELL_KEYS})

    known_section = {k: v for k, v in section.items() if k in _SECTION_KEYS}
    known_section.pop('thresholds', None)
    known_section.pop('global', None)
    known_section.pop('wells', None)

    return BywellConfig(data_output_dir=config['data_output_dir'],
                    local_output_dir=config['local_output_dir'],
                    thresholds=_build_thresholds(section, config),
                    schema=_build_schema(config),
                    wells=wells,
                    defaults=defaults,
                    **known_section)


def load_allwells_config(source: Union[str, Dict[str, Any], AllwellsConfig]) -> AllwellsConfig:
    """Build an :class:`AllwellsConfig` from a file path or a mapping.

    Only the across-wells section is checked here; the top level is the
    concern of :func:`load_bywell_config`, which would otherwise report the
    same keys twice.
    """
    if isinstance(source, AllwellsConfig):
        return source

    config = load_config_from_file(source) if isinstance(source, str) else source
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping, got {type(config).__name__}")

    section = config.get(ALLWELLS_SECTION)
    if section is None:
        # A flat configuration, or one that simply leaves these at their
        # defaults.  Either way there is no section to check.
        section = {}
    elif not isinstance(section, dict):
        raise ValueError(f"'{ALLWELLS_SECTION}' is present but holds no settings; "
                         f"remove it or fill it in")
    else:
        _warn_unknown(section, _ALLWELLS_KEYS, _LATER_STEP_ALLWELLS, ALLWELLS_SECTION)

    return AllwellsConfig(data_output_dir=config.get('data_output_dir', ''),
                          local_output_dir=config.get('local_output_dir', ''),
                          schema=_build_schema(config),
                          **{k: v for k, v in section.items() if k in _ALLWELLS_KEYS})


def load_guide_config(source: Union[str, Dict[str, Any], GuideConfig]) -> GuideConfig:
    """Build a :class:`GuideConfig` from a file path or a mapping.

    Only the guide filtering section is checked here, for the same reason
    as :func:`load_allwells_config`.
    """
    if isinstance(source, GuideConfig):
        return source

    config = load_config_from_file(source) if isinstance(source, str) else source
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping, got {type(config).__name__}")

    section = config.get(GUIDE_SECTION)
    if section is None:
        section = {}
    elif not isinstance(section, dict):
        raise ValueError(f"'{GUIDE_SECTION}' is present but holds no settings; "
                         f"remove it or fill it in")
    else:
        _warn_unknown(section, _GUIDE_KEYS, frozenset(), GUIDE_SECTION)

    return GuideConfig(data_output_dir=config.get('data_output_dir', ''),
                       local_output_dir=config.get('local_output_dir', ''),
                       schema=_build_schema(config),
                       **{k: v for k, v in section.items() if k in _GUIDE_KEYS})


def load_preprocessing_config(
        source: Union[str, Dict[str, Any], BywellConfig, PreprocessingConfig]
) -> PreprocessingConfig:
    """Build a :class:`PreprocessingConfig` from a file path or a mapping.

    A :class:`BywellConfig` is also accepted, for callers that only run the
    by-well steps.  There is no mapping left to read the other sections
    from in that case, so those settings take their defaults.
    """
    if isinstance(source, PreprocessingConfig):
        return source

    if isinstance(source, BywellConfig):
        return PreprocessingConfig(
            bywell=source,
            allwells=AllwellsConfig(data_output_dir=source.data_output_dir,
                                    local_output_dir=source.local_output_dir),
            guides=GuideConfig(data_output_dir=source.data_output_dir,
                               local_output_dir=source.local_output_dir)
        )

    config = load_config_from_file(source) if isinstance(source, str) else source
    return PreprocessingConfig(bywell=load_bywell_config(config),
                               allwells=load_allwells_config(config),
                               guides=load_guide_config(config))
