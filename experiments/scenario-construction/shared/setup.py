"""Shared setup boilerplate for scenario construction approaches.

Consolidates argument parsing and config loading across approach_*.py files.
"""

import argparse
from pathlib import Path
from dataclasses import dataclass

from .config import get_target, TARGETS
from .signals import DEFAULT_MAX_HORIZON_DAYS, compute_signal_cutoff


@dataclass
class ApproachConfig:
    """Configuration for a scenario construction approach."""

    # From target config
    target: str
    question_text: str
    context: str
    question_type: str  # "continuous" or "binary"
    config: object  # Full config object
    target_resolution_date: object | None  # date object or None

    # Paths
    repo_root: Path
    db_path: Path
    output_dir: Path

    # Settings
    voi_floor: float
    max_horizon_days: int
    knowledge_cutoff: str | None = None

    # Optional (set by specific approaches)
    n_uncertainties: int | None = None
    match_threshold: float | None = None


# Standard prediction market sources
DEFAULT_SOURCES = ["polymarket", "metaculus", "kalshi", "infer", "manifold"]


def create_base_parser(description: str) -> argparse.ArgumentParser:
    """Create argument parser with common arguments.

    Args:
        description: Description for the parser

    Returns:
        ArgumentParser with --target and --voi-floor arguments
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--target",
        choices=list(TARGETS.keys()),
        default="gdp_2050",
        help="Target question to generate scenarios for",
    )
    parser.add_argument(
        "--voi-floor",
        type=float,
        default=0.1,
        help="VOI floor for scenario generation (default 0.1)",
    )
    return parser


def add_uncertainty_args(parser: argparse.ArgumentParser) -> None:
    """Add uncertainty-related arguments to parser.

    Args:
        parser: ArgumentParser to extend
    """
    parser.add_argument(
        "--n-uncertainties",
        type=int,
        default=3,
        help="Number of uncertainty axes to identify",
    )


def add_matching_args(parser: argparse.ArgumentParser) -> None:
    """Add signal matching arguments to parser.

    Args:
        parser: ArgumentParser to extend
    """
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.6,
        help="Similarity threshold for matching top-down to market signals (default 0.6)",
    )


def load_config(args, script_path: Path) -> ApproachConfig:
    """Load configuration from parsed arguments.

    Args:
        args: Parsed argument namespace
        script_path: Path to the calling script (__file__)

    Returns:
        ApproachConfig with all settings populated
    """
    config = get_target(args.target)

    # Compute paths relative to script location
    repo_root = script_path.parent.parent.parent
    db_path = repo_root / "data" / "forecastbench.db"
    output_dir = script_path.parent / "results" / args.target
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map QuestionType enum to string for generate_mece_scenarios
    question_type = config.question.question_type.value  # "continuous" or "binary"

    # Get target resolution date and compute smart cutoff
    target_resolution_date = config.question.resolution_date
    max_horizon_days = compute_signal_cutoff(target_resolution_date, DEFAULT_MAX_HORIZON_DAYS)

    return ApproachConfig(
        target=args.target,
        question_text=config.question.text,
        context=config.context,
        question_type=question_type,
        config=config,
        target_resolution_date=target_resolution_date,
        repo_root=repo_root,
        db_path=db_path,
        output_dir=output_dir,
        voi_floor=args.voi_floor,
        max_horizon_days=max_horizon_days,
        knowledge_cutoff=getattr(args, "knowledge_cutoff", None),
        n_uncertainties=getattr(args, "n_uncertainties", None),
        match_threshold=getattr(args, "match_threshold", None),
    )


def print_header(
    approach_name: str,
    target: str,
    sources: list[str] | None = None,
    max_horizon_days: int | None = None,
    n_uncertainties: int | None = None,
    voi_floor: float | None = None,
    match_threshold: float | None = None,
) -> None:
    """Print standard approach header.

    Args:
        approach_name: Name of the approach (e.g., "BOTTOM-UP")
        target: Target question text
        sources: List of sources (optional)
        max_horizon_days: Max days until resolution for actionable signals (optional)
        n_uncertainties: Number of uncertainties (optional)
        voi_floor: VOI floor threshold (optional)
        match_threshold: Match threshold (optional)
    """
    print("=" * 60)
    print(f"{approach_name} APPROACH v7: Enhanced Data Model")
    print("=" * 60)
    print(f"\nTarget: {target}")

    if sources:
        print(f"Sources: {', '.join(sources)}")
    if max_horizon_days is not None:
        print(f"Signal horizon: {max_horizon_days} days")
    if n_uncertainties is not None:
        print(f"Uncertainty axes: {n_uncertainties}")
    if voi_floor is not None:
        print(f"VOI floor: {voi_floor}")
    if match_threshold is not None:
        print(f"Match threshold: {match_threshold}")
