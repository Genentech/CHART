"""
The archetype analysis stage.

All but the first step are R scripts, so this module mostly does not
implement them: it resolves the paths they need from the configuration
and calls them, once per channel and, where it matters, once per
gamma/k.knn pair.  The pca step that builds their input is CHART's own,
and runs in process.

Each script is told which directory to work in through the
:data:`ENV_DIR` environment variable, and where to leave diagnostics
through :data:`ENV_REPORTS`.  Both are read with a fallback, so the
scripts stay runnable by hand with ``export``, and their argument lists
stay as they already are.
"""

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from ..io.config import ARCHETYPES_SECTION, ArchetypesConfig, load_archetypes_config

logger = logging.getLogger(__name__)

#: How the per-channel directory reaches the R scripts.
ENV_DIR = 'CHART_ARCHETYPE_DIR'

#: Where they put the diagnostics nothing downstream reads.  Left unset,
#: each script falls back to writing them beside the data.
ENV_REPORTS = 'CHART_ARCHETYPE_REPORT_DIR'

#: The script behind each step, relative to the scripts directory.
SCRIPTS = {
    'npc': 'pick_npc.R',
    'metacells': 'metacell_adaptive_npc.R',
    'evaluate': 'eval_spaces_perchannel.R',
    'archetypes': 'robustAA/robust_archetyping.R',
    'selectk': 'robustAA/test_evalK_onfullPC.R',
    'weights': 'robustAA/archetype_weights.R',
}

#: Steps CHART runs itself rather than handing to an R script.
PYTHON_STEPS = ('pca',)

#: The steps, in the order they have to run.
STEPS = PYTHON_STEPS + tuple(SCRIPTS)

#: Steps that run once per gamma/k.knn pair rather than once per channel.
PER_PARAMS = ('metacells', 'archetypes')

#: Steps that work from the one chosen pair, and take the same arguments.
SELECTED_PAIR = ('selectk', 'weights')

#: Everything the scripts load, checked before the first one runs.  They
#: are installed together by environment.yml, and the steps share most of
#: them, so the whole set is asked for rather than a set per step.
R_PACKAGES = ('SuperCell', 'archetypes', 'arrow', 'FNN', 'quadprog', 'clue',
              'dplyr', 'ggplot2')


@dataclass
class StepResult:
    """One invocation of one script."""
    step: str
    channel: str
    command: List[str]
    directory: str
    reports: str = ''
    returncode: Optional[int] = None

    @property
    def failed(self) -> bool:
        return self.returncode not in (0, None)


def scripts_dir() -> Path:
    """Where the R scripts live.

    They ship inside the package, so this resolves the same way in a
    checkout and in an install.  ``--scripts-dir`` overrides it, for
    working on a copy of the scripts without reinstalling.
    """
    return Path(__file__).resolve().parent / 'scripts'


def _arguments(step: str, config: ArchetypesConfig, channel: str,
               pair: Optional[Sequence[Any]]) -> List[str]:
    """The arguments one step wants, in the order its script reads them."""
    directory = config.channel_path(channel)
    npc = config.npc.get(channel)

    if step == 'npc':
        # The only step given explicit paths: it is the one a workflow runs
        # to learn npc before it can name anything else's output files.
        arguments = [f'{directory}/cell_pca_centered.parquet',
                     f'{directory}/npc.txt']
        return arguments + ([str(npc)] if npc is not None else [])
    if step == 'metacells':
        arguments = [str(pair[0]), str(pair[1]), channel]
        return arguments + ([str(npc)] if npc is not None else [])
    if step == 'archetypes':
        return [str(pair[0]), str(pair[1]), channel, 'FALSE']
    if step in SELECTED_PAIR:
        # Which swept pair carries on, and whether K was decided already.
        # Left out, selectk picks it and weights reads what it picked.
        gamma, knn = config.selected
        arguments = [channel, 'FALSE', str(gamma), str(knn)]
        k = config.k.get(channel)
        return arguments + ([str(k)] if k is not None else [])
    return [channel]


def _run_pca(config: ArchetypesConfig, channel: str) -> str:
    """Build the space the rest of the stage reads, for one channel."""
    from ..dimred.channels import write_channel_pca

    if not config.all_channels:
        raise ValueError(
            f"The pca step needs every channel the feature table holds, "
            f"because it finds one channel's columns by removing the "
            f"others'. Name them under '{ARCHETYPES_SECTION}.all_channels'.")

    return write_channel_pca(config.features_path, channel,
                             config.all_channels,
                             config.channel_path(channel), config.schema)


def _check_r() -> None:
    """Say what R is missing before the first step, not partway through.

    Otherwise a package the last step needs is reported only once the
    steps before it have spent their hours.
    """
    if not shutil.which('Rscript'):
        raise RuntimeError("Rscript is not on PATH, and all but the pca step "
                           "are R scripts; create the environment described "
                           "in environment.yml, ask for --steps pca, or run "
                           "with --dry-run")

    # Loaded rather than looked for on disk, because r-supercell and the
    # other compiled packages are built against one R minor version, so a
    # package can be installed and still not load.
    names = ', '.join(f'"{package}"' for package in R_PACKAGES)
    probe = (f'cat(Filter(function(p) !suppressWarnings(suppressMessages('
             f'require(p, character.only = TRUE, quietly = TRUE))), '
             f'c({names})), sep = " ")')
    asked = subprocess.run(['Rscript', '-e', probe],
                           capture_output=True, text=True)
    if asked.returncode != 0:
        raise RuntimeError(f"Rscript is on PATH but could not be asked what "
                           f"it has installed:\n{asked.stderr.strip()}")

    missing = asked.stdout.split()
    if missing:
        raise RuntimeError(f"R cannot load {', '.join(missing)}, which the "
                           f"archetype scripts need; the environment "
                           f"described in environment.yml has every package "
                           f"they load")


def _planned(config: ArchetypesConfig, channels: Sequence[str],
             steps: Sequence[str], directory: Path) -> List[StepResult]:
    """Work out every invocation before running any of it."""
    planned = []
    for channel in channels:
        for step in steps:
            if step in PYTHON_STEPS:
                planned.append(StepResult(
                    step=step, channel=channel,
                    directory=config.channel_path(channel),
                    reports=config.report_path(channel),
                    command=[f'{step} [in CHART]', channel]))
                continue

            script = directory / SCRIPTS[step]
            if not script.exists():
                raise FileNotFoundError(
                    f"No script for the {step} step at {script}; name the "
                    f"directory holding the R scripts with --scripts-dir")

            pairs: List[Optional[Sequence[Any]]] = [None]
            if step in PER_PARAMS:
                if not config.params:
                    raise ValueError(
                        f"The {step} step runs once per gamma and k.knn pair, "
                        f"and '{ARCHETYPES_SECTION}.params' names none")
                pairs = list(config.params)

            for pair in pairs:
                planned.append(StepResult(
                    step=step, channel=channel,
                    directory=config.channel_path(channel),
                    reports=config.report_path(channel),
                    command=['Rscript', str(script)]
                            + _arguments(step, config, channel, pair)))
    return planned


def run_archetypes(config: Union[str, Dict[str, Any], ArchetypesConfig],
                   channels: Optional[Sequence[str]] = None,
                   steps: Sequence[str] = ('all',),
                   scripts: Optional[str] = None,
                   dry_run: bool = False,
                   continue_on_error: bool = False) -> List[StepResult]:
    """Run the archetype analysis steps.

    Args:
        config: An :class:`~chart.io.config.ArchetypesConfig`, or a
                configuration file or mapping to build one from
        channels: Channels to run; ``None`` runs every channel in the config
        steps: Which of :data:`STEPS` to run; ``('all',)`` runs them all
        scripts: Where the R scripts are; ``None`` looks beside the package
        dry_run: Report the commands without running them
        continue_on_error: Log and carry on when a script fails

    Returns:
        One :class:`StepResult` per invocation, in the order planned.
    """
    config = load_archetypes_config(config)
    channels = list(channels) if channels else list(config.channels)
    if not channels:
        raise ValueError(f"No channels to run: name them with --channel or "
                         f"under '{ARCHETYPES_SECTION}.channels'")

    wanted = STEPS if 'all' in steps else tuple(s for s in STEPS if s in steps)
    planned = _planned(config, channels, wanted,
                       Path(scripts) if scripts else scripts_dir())

    needs_r = any(result.step not in PYTHON_STEPS for result in planned)
    if needs_r and not dry_run:
        _check_r()

    logger.info(f"Archetypes: {len(planned)} invocations across "
                f"{len(channels)} channels")
    for result in planned:
        logger.info(f"{result.step} [{result.channel}]: "
                    f"{' '.join(result.command)}")
        if dry_run:
            continue

        if result.step in PYTHON_STEPS:
            try:
                _run_pca(config, result.channel)
            except Exception as error:
                if not continue_on_error:
                    raise
                result.returncode = 1
                logger.error(f"The {result.step} step failed for channel "
                             f"{result.channel}: {error}")
                continue
            result.returncode = 0
            continue

        completed = subprocess.run(result.command,
                                   env={**os.environ,
                                        ENV_DIR: result.directory,
                                        ENV_REPORTS: result.reports})
        result.returncode = completed.returncode
        if result.failed:
            if not continue_on_error:
                raise RuntimeError(
                    f"The {result.step} step failed for channel "
                    f"{result.channel} (exit {result.returncode}); its own "
                    f"output above says why")
            logger.error(f"The {result.step} step failed for channel "
                         f"{result.channel} (exit {result.returncode})")

    failed = [r for r in planned if r.failed]
    logger.info(f"Archetypes complete: {len(planned) - len(failed)}/"
                f"{len(planned)} invocations succeeded")
    return planned
