"""
Command-line entry point.

This module parses arguments, configures logging handlers, and
turns a run into an exit status.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from typing import List, Optional, Sequence

from .io.config import join_path, load_bywell_config
from .filtering import TARGETS
from .preprocessing import STEPS, run_preprocessing
from .qc import COMPONENTS

logger = logging.getLogger(__name__)

_HANDLER_TAG = '_chart_cli_handler'


def setup_logging(log_dir: str, name: str, level: int = logging.INFO) -> str:
    """Send log records to a timestamped file in *log_dir*, and to stdout.

    Returns the path of the log file.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_qc_log_{timestamp}.txt")

    root = logging.getLogger()
    root.setLevel(level)

    # Handlers are attached directly rather than through basicConfig, which
    # does nothing once the root logger has any handler.  Handlers from an
    # earlier call are dropped so a second call does not keep writing to the
    # first call's file.
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            root.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in (logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        setattr(handler, _HANDLER_TAG, True)
        root.addHandler(handler)

    return log_file


def build_parser() -> argparse.ArgumentParser:
    """Build the ``chart`` argument parser."""
    parser = argparse.ArgumentParser(
        prog='chart',
        description="CHART: Cellular High-content imaging Archetype Response Toolkit")
    subcommands = parser.add_subparsers(dest='command', required=True,
                                        metavar='<command>')
    _add_preprocessing_parser(subcommands)
    return parser


def _add_preprocessing_parser(subcommands) -> None:
    """Add the ``chart preprocessing`` subcommand."""
    parser = subcommands.add_parser(
        'preprocessing',
        help='Run the by-well preprocessing pipeline',
        description='Run the by-well preprocessing pipeline on one well or all of them.')

    parser.add_argument('--config', required=True,
                        help='Path to configuration file (YAML or JSON)')

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--well', help='Well identifier to process')
    target.add_argument('--all-wells', action='store_true',
                        help='Process every well defined in the configuration')

    parser.add_argument('--steps', nargs='+',
                        choices=('all',) + STEPS,
                        default=['all'],
                        help='Pipeline steps to run (default: all)')
    parser.add_argument('--components', nargs='+',
                        choices=('all',) + COMPONENTS,
                        default=['all'],
                        help='For the qc step, which components to run '
                             '(default: all). Naming a component explicitly '
                             'makes its prerequisites mandatory, so the run '
                             'fails rather than skipping it when they are '
                             'missing.')
    parser.add_argument('--targets', nargs='+',
                        choices=('all',) + TARGETS,
                        default=['all'],
                        help='For the filter step, what to filter '
                             '(default: all)')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save plots to the output directory')
    parser.add_argument('--plots-dir',
                        help='Directory to save plots (default: {qc_dir}/plots)')
    parser.add_argument('--continue-on-error', action='store_true',
                        help='Carry on when a well fails, instead of stopping')
    parser.add_argument('--quiet', action='store_true',
                        help='Log warnings and errors only')

    deprecated = parser.add_argument_group('deprecated')
    deprecated.add_argument('--skip-if', action='store_true',
                            help='Skip pheno correlation analysis (use --components)')
    deprecated.add_argument('--skip-iss-if', action='store_true',
                            help='Skip pheno-SBS correlation analysis (use --components)')

    parser.set_defaults(func=_preprocessing_command)


def resolve_components(args: argparse.Namespace) -> List[str]:
    """Translate the deprecated skip flags into a component list."""
    if not (args.skip_if or args.skip_iss_if):
        return list(args.components)

    excluded = set()
    if args.skip_if:
        excluded.add('pheno')
    if args.skip_iss_if:
        excluded.add('pheno_sbs')
    components = [c for c in COMPONENTS if c not in excluded]

    print(f"Warning: --skip-if and --skip-iss-if are deprecated, use --components. "
          f"Running: {' '.join(components)}", file=sys.stderr)
    return components


def _preprocessing_command(args: argparse.Namespace) -> int:
    """Configure logging and hand over to the preprocessing workflow."""
    config = load_bywell_config(args.config)
    
    log_dir = join_path(config.local_output_dir, config.qc_dir)
    log_file = setup_logging(log_dir, args.well or 'all_wells',
                             level=logging.WARNING if args.quiet else logging.INFO)
    logger.info(f"Logging to {log_file}")

    try:
        result = run_preprocessing(config,
                                   wells=None if args.all_wells else [args.well],
                                   steps=args.steps,
                                   components=resolve_components(args),
                                   targets=args.targets,
                                   save_plots=args.save_plots,
                                   plots_dir=args.plots_dir,
                                   continue_on_error=args.continue_on_error)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        return 1

    return 1 if result.failed else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run CHART from the command line.  Returns a process exit status."""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
