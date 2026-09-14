"""
The filter-list contract between QC and filtering.

QC writes per-well lists of labels; filtering applies them.  A manifest
written alongside those lists records what was produced, so that the
reader does not have to infer meaning from filenames and can tell a
filter that was never produced from one that excluded nobody.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import List, Optional

from .. import __version__

logger = logging.getLogger(__name__)

#: A cell has to appear in every inclusion filter to survive.
INCLUSION = 'inclusion'
#: A cell appearing in any exclusion filter is removed.
EXCLUSION = 'exclusion'
ROLES = (INCLUSION, EXCLUSION)

MANIFEST_FILE = '{well}_filters.json'

#: Filters written before manifests existed, by cellmapp and by early
#: CHART.  Used to reconstruct a manifest for a directory without one.
LEGACY_FILTERS = (
    ('phenocorfilt', INCLUSION, '{well}_phenocorfilt.parquet'),
    ('phenosbscorfilt', INCLUSION, '{well}_phenosbscorfilt.parquet'),
    ('high_ratio_segmentation_errors', EXCLUSION,
     '{well}_high_ratio_segmentation_errors.parquet'),
)


@dataclass
class FilterEntry:
    """One filter list: what it is called, what it means, where it is."""
    name: str
    role: str
    file: str
    labels: int

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(f"Filter '{self.name}' has role {self.role!r}; "
                             f"expected one of {', '.join(ROLES)}")


@dataclass
class FilterManifest:
    """What QC produced for one well."""
    well: str
    created: str = ''
    chart_version: str = ''
    components_run: List[str] = field(default_factory=list)
    components_skipped: List[str] = field(default_factory=list)
    filters: List[FilterEntry] = field(default_factory=list)

    def by_role(self, role: str) -> List[FilterEntry]:
        return [entry for entry in self.filters if entry.role == role]

    def save(self, filter_dir: str) -> str:
        """Write the manifest, stamping it with the time and version."""
        os.makedirs(filter_dir, exist_ok=True)
        self.created = datetime.now().isoformat(timespec='seconds')
        self.chart_version = __version__
        path = os.path.join(filter_dir, MANIFEST_FILE.format(well=self.well))
        with open(path, 'w') as handle:
            json.dump(asdict(self), handle, indent=2)
        logger.info(f"Saved filter manifest: {path} ({len(self.filters)} filters)")
        return path


def load_manifest(filter_dir: str, well: str) -> Optional[FilterManifest]:
    """Read the manifest for *well*, or ``None`` when there is not one."""
    path = os.path.join(filter_dir, MANIFEST_FILE.format(well=well))
    if not os.path.exists(path):
        return None

    with open(path) as handle:
        raw = json.load(handle)

    entries = [FilterEntry(name=entry['name'], role=entry['role'],
                           file=entry['file'], labels=entry['labels'])
               for entry in raw.get('filters', [])]
    return FilterManifest(well=raw.get('well', well),
                          created=raw.get('created', ''),
                          chart_version=raw.get('chart_version', ''),
                          components_run=raw.get('components_run', []),
                          components_skipped=raw.get('components_skipped', []),
                          filters=entries)


def manifest_from_directory(filter_dir: str, well: str) -> FilterManifest:
    """Reconstruct a manifest from the filter files that happen to be present.

    For filter directories written before manifests existed.  Only the
    filters in :data:`LEGACY_FILTERS` can be recognised this way, which is
    why anything else needs a manifest to be seen at all.
    """
    manifest = FilterManifest(well=well)
    for name, role, template in LEGACY_FILTERS:
        filename = template.format(well=well)
        if os.path.exists(os.path.join(filter_dir, filename)):
            # The label count is not known without reading the file; the
            # reader fills it in.
            manifest.filters.append(FilterEntry(name=name, role=role,
                                                file=filename, labels=-1))
    return manifest
