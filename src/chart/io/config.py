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
import yaml
from dataclasses import dataclass, field, fields, replace
from typing import Any, Dict, List, Optional, Union

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
_OTHER_STAGE_TOP_LEVEL = frozenset({'preprocessing_allwells', 'guide_filtering'})
_OTHER_STAGE_SECTION = frozenset({'barcode_file', 'labels', 'feature_types',
                                  'filtered_output_dir'})
_OTHER_STAGE_WELL = frozenset({'feature_dir', 'reads_file_pattern', 'reads_file',
                               'filter_duplicates', 'barcode_colname'})

_TOP_LEVEL_KEYS = frozenset({'data_output_dir', 'local_output_dir', SECTION_NAME})
_SECTION_KEYS = frozenset({'scallops_dir', 'merged_dir', 'filter_dir', 'qc_dir',
                           'premerged', 'column_mapping', 'precomputed_filters',
                           'thresholds', 'global', 'wells'})


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
    """The settings CHART's QC reads, with their defaults in one place."""
    data_output_dir: str
    local_output_dir: str
    scallops_dir: str = ''
    merged_dir: str = 'preprocessing/bywell/'
    filter_dir: str = 'preprocessing/bywell/filters/'
    qc_dir: str = 'preprocessing/bywell/qc_reports/'
    premerged: bool = False
    column_mapping: Optional[Dict[str, str]] = None
    precomputed_filters: Optional[Dict[str, Any]] = None
    thresholds: Thresholds = field(default_factory=Thresholds)
    wells: Dict[str, WellConfig] = field(default_factory=dict)
    defaults: WellConfig = field(default_factory=WellConfig)

    def well(self, name: str) -> WellConfig:
        """Settings for *name*, falling back to the global block when the
        well is not listed in the configuration."""
        if name in self.wells:
            return self.wells[name]
        return replace(self.defaults, well=name)

    def well_names(self) -> List[str]:
        """Every well named in the configuration."""
        if not self.wells:
            raise ValueError(
                f"Configuration must contain a 'wells' section under "
                f"'{SECTION_NAME}' to process all wells"
            )
        return list(self.wells)


_THRESHOLD_KEYS = frozenset(f.name for f in fields(Thresholds))
_WELL_KEYS = frozenset(f.name for f in fields(WellConfig)) - {'well'}


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
                      _OTHER_STAGE_TOP_LEVEL | _OTHER_STAGE_SECTION, 'the top level')

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
                    wells=wells,
                    defaults=defaults,
                    **known_section)
