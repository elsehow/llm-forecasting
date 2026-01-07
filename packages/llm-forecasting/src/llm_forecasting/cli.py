"""Command-line interface for ForecastBench.

Commands:
    update-questions    Fetch questions from all sources and update the database
    create-question-set Create a new question set for evaluation
    forecast            Generate forecasts for a question set
    resolve             Check resolutions and compute scores
    leaderboard         View current standings
"""

import asyncio
import json
import logging
from datetime import date, timedelta

import click

from llm_forecasting.config import settings
from llm_forecasting.models import Question, Resolution, SourceType
from llm_forecasting.resolution import resolve_question, score_forecast
from llm_forecasting.sampling import QuestionSampler, SamplingConfig
from llm_forecasting.eval.scoring import build_leaderboard, compute_pairwise_significance, format_leaderboard
from llm_forecasting.sources import registry
from llm_forecasting.storage.sqlite import SQLiteStorage

# Default evaluation horizons (days from forecast_due_date)
DEFAULT_HORIZONS = [7, 14, 30, 90, 180, 365]

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_storage() -> SQLiteStorage:
    """Get storage instance from settings."""
    # Extract path from database URL
    db_url = settings.database_url
    if db_url.startswith("sqlite+aiosqlite:///"):
        db_path = db_url.replace("sqlite+aiosqlite:///", "")
    else:
        db_path = "forecastbench.db"
    return SQLiteStorage(db_path)


@click.group()
@click.version_option(version="2.0.0")
def main():
    """ForecastBench - A benchmark for LLM forecasting accuracy."""
    pass


@main.command()
@click.option(
    "--source",
    "-s",
    multiple=True,
    help="Specific source(s) to update. Default: all sources.",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output.")
def update_questions(source: tuple[str, ...], verbose: bool):
    """Fetch questions from all sources and update the database.

    This command:
    1. Fetches current questions from each source
    2. Updates the database with new questions and changes
    3. Records current prices/probabilities for resolution
    """
    asyncio.run(_update_questions(source, verbose))


async def _update_questions(sources: tuple[str, ...], verbose: bool):
    """Async implementation of update-questions command."""
    storage = get_storage()

    # Determine which sources to update
    if sources:
        source_names = list(sources)
        for name in source_names:
            if name not in registry:
                click.echo(f"Unknown source: {name}. Available: {registry.list()}", err=True)
                return
    else:
        source_names = registry.list()

    click.echo(f"Updating questions from: {', '.join(source_names)}")

    total_questions = 0
    total_resolutions = 0
    today = date.today()

    for source_name in source_names:
        click.echo(f"\n📡 Fetching from {source_name}...")

        try:
            source_cls = registry.get(source_name)
            source = source_cls()
            questions = await source.fetch_questions()

            click.echo(f"   Found {len(questions)} questions")

            # Save questions
            await storage.save_questions(questions)
            total_questions += len(questions)

            # Record resolution values (current prices/probabilities)
            resolution_count = 0
            for q in questions:
                if q.base_rate is not None:
                    resolution = Resolution(
                        question_id=q.id,
                        source=q.source,
                        date=today,
                        value=q.base_rate,
                    )
                    await storage.save_resolution(resolution)
                    resolution_count += 1

            total_resolutions += resolution_count

            if verbose:
                click.echo(f"   Recorded {resolution_count} resolution values")

        except Exception as e:
            click.echo(f"   ❌ Error: {e}", err=True)
            logger.exception(f"Error fetching from {source_name}")

    click.echo(f"\n✅ Updated {total_questions} questions, recorded {total_resolutions} values")
    await storage.close()


@main.command()
@click.option(
    "--name",
    "-n",
    default=None,
    help="Name for the question set. Default: YYYY-MM-DD-llm",
)
@click.option(
    "--num-questions",
    "-q",
    default=500,
    help="Number of questions to sample. Default: 500",
)
@click.option(
    "--freeze-window",
    "-w",
    default=10,
    help="Days between freeze and forecast due date. Default: 10",
)
@click.option(
    "--horizons",
    "-h",
    default=",".join(map(str, DEFAULT_HORIZONS)),
    help=f"Comma-separated evaluation horizons (days). Default: {','.join(map(str, DEFAULT_HORIZONS))}",
)
@click.option("--dry-run", is_flag=True, help="Show what would be created without saving.")
def create_question_set(
    name: str | None,
    num_questions: int,
    freeze_window: int,
    horizons: str,
    dry_run: bool,
):
    """Create a new question set for evaluation.

    This command:
    1. Samples questions from the database (stratified by source, category, base rate)
    2. Sets freeze_date (today) and forecast_due_date (freeze_date + window)
    3. Sets resolution dates at each horizon
    """
    asyncio.run(_create_question_set(name, num_questions, freeze_window, horizons, dry_run))


async def _create_question_set(
    name: str | None,
    num_questions: int,
    freeze_window: int,
    horizons_str: str,
    dry_run: bool,
):
    """Async implementation of create-question-set command."""
    storage = get_storage()

    # Parse horizons
    horizon_days = [int(h.strip()) for h in horizons_str.split(",")]

    # Set dates
    freeze_date = date.today()
    forecast_due_date = freeze_date + timedelta(days=freeze_window)
    resolution_dates = [forecast_due_date + timedelta(days=h) for h in horizon_days]

    # Generate name if not provided
    if name is None:
        name = f"{freeze_date.isoformat()}-llm"

    # Check if name already exists
    existing = await storage.get_question_set_by_name(name)
    if existing:
        click.echo(f"❌ Question set '{name}' already exists", err=True)
        await storage.close()
        return

    click.echo(f"Creating question set: {name}")
    click.echo(f"  Freeze date: {freeze_date}")
    click.echo(f"  Forecast due date: {forecast_due_date}")
    click.echo(f"  Resolution horizons: {horizon_days} days")
    click.echo(f"  Sampling {num_questions} questions...")

    # Fetch all questions from database
    all_questions = await storage.get_questions(resolved=False)
    click.echo(f"  Found {len(all_questions)} unresolved questions in database")

    if len(all_questions) == 0:
        click.echo("❌ No questions in database. Run 'update-questions' first.", err=True)
        await storage.close()
        return

    # Sample questions
    config = SamplingConfig(num_questions=num_questions)
    sampler = QuestionSampler(config)
    result = sampler.sample_stratified(all_questions, reference_date=freeze_date)

    click.echo(f"\n  Sampled {len(result.questions)} questions:")
    click.echo(f"    By source: {result.source_counts}")
    click.echo(f"    By category: {result.category_counts}")
    click.echo(f"    By base rate: {result.base_rate_distribution}")

    if result.warnings:
        for warning in result.warnings:
            click.echo(f"    ⚠️  {warning}")

    if dry_run:
        click.echo("\n🔍 Dry run - not saving")
        await storage.close()
        return

    # Create the question set
    question_set_id = await storage.create_question_set(
        name=name,
        freeze_date=freeze_date,
        forecast_due_date=forecast_due_date,
        resolution_dates=resolution_dates,
        questions=result.questions,
    )

    click.echo(f"\n✅ Created question set #{question_set_id}: {name}")
    await storage.close()


@main.command()
@click.option(
    "--model",
    "-m",
    default=None,
    help=f"Model to use. Default: {settings.default_model}",
)
@click.option(
    "--question-set",
    "-q",
    type=int,
    default=None,
    help="Question set ID to forecast. Default: latest pending set.",
)
@click.option("--dry-run", is_flag=True, help="Show what would be done without forecasting.")
def forecast(model: str | None, question_set: int | None, dry_run: bool):
    """Generate forecasts for a question set.

    Uses the specified LLM (via LiteLLM) to forecast all questions in the set.
    """
    asyncio.run(_forecast(model, question_set, dry_run))


async def _forecast(model: str | None, question_set_id: int | None, dry_run: bool):
    """Async implementation of forecast command."""
    from llm_forecasting.agents.llm import LLMForecaster

    storage = get_storage()

    # Get the question set
    if question_set_id is None:
        # Get latest pending/forecasting set
        sets = await storage.get_question_sets()
        pending_sets = [s for s in sets if s["status"] in ("pending", "forecasting")]
        if not pending_sets:
            click.echo("❌ No pending question sets to forecast", err=True)
            await storage.close()
            return
        qs = pending_sets[0]
        question_set_id = qs["id"]
    else:
        qs = await storage.get_question_set(question_set_id)
        if not qs:
            click.echo(f"❌ Question set #{question_set_id} not found", err=True)
            await storage.close()
            return

    click.echo(f"Forecasting question set #{question_set_id}: {qs['name']}")
    click.echo(f"  Status: {qs['status']}")
    click.echo(f"  Forecast due date: {qs['forecast_due_date']}")

    # Check if past due date
    if date.today() < qs["forecast_due_date"]:
        click.echo(f"  ⚠️  Warning: Forecast due date is in the future")

    # Get questions
    items = await storage.get_question_set_items(question_set_id)
    click.echo(f"  Questions: {len(items)}")

    # Initialize forecaster
    model_name = model or settings.default_model
    click.echo(f"  Model: {model_name}")

    if dry_run:
        click.echo("\n🔍 Dry run - not forecasting")
        await storage.close()
        return

    forecaster = LLMForecaster(model=model_name)

    # Update status
    await storage.update_question_set_status(question_set_id, "forecasting")

    # Forecast each question
    success_count = 0
    error_count = 0

    with click.progressbar(items, label="Forecasting") as bar:
        for item in bar:
            try:
                # Get the full question
                question = await storage.get_question(item["source"], item["question_id"])
                if not question:
                    logger.warning(f"Question not found: {item['source']}/{item['question_id']}")
                    error_count += 1
                    continue

                # Generate forecast
                fc = await forecaster.forecast(question)

                # Save with question_set_id
                await storage.save_forecast(fc, question_set_id=question_set_id)
                success_count += 1

            except Exception as e:
                logger.error(f"Error forecasting {item['question_id']}: {e}")
                error_count += 1

    click.echo(f"\n✅ Generated {success_count} forecasts ({error_count} errors)")
    await storage.close()


@main.command()
@click.option(
    "--question-set",
    "-q",
    type=int,
    default=None,
    help="Question set ID to resolve. Default: all sets needing resolution.",
)
def resolve(question_set: int | None):
    """Check resolutions and compute scores.

    For each question set:
    1. Check if questions have resolved (markets) or fetch current values (data)
    2. Compute Brier scores at each evaluation horizon
    3. Update the leaderboard
    """
    asyncio.run(_resolve(question_set))


async def _resolve(question_set_id: int | None):
    """Async implementation of resolve command."""
    storage = get_storage()

    # Get question sets to resolve
    if question_set_id is not None:
        qs = await storage.get_question_set(question_set_id)
        if not qs:
            click.echo(f"❌ Question set #{question_set_id} not found", err=True)
            await storage.close()
            return
        question_sets = [qs]
    else:
        # Get all sets that need resolution (have forecasts, not completed)
        all_sets = await storage.get_question_sets()
        question_sets = [s for s in all_sets if s["status"] in ("forecasting", "resolving")]

    if not question_sets:
        click.echo("No question sets to resolve")
        await storage.close()
        return

    today = date.today()
    total_scores = 0
    skipped_count = 0

    for qs in question_sets:
        click.echo(f"\n📊 Resolving question set #{qs['id']}: {qs['name']}")

        # Parse resolution dates
        resolution_dates = [date.fromisoformat(d) for d in qs["resolution_dates"]]
        forecast_due_date = qs["forecast_due_date"]

        # Get items and forecasts
        items = await storage.get_question_set_items(qs["id"])
        click.echo(f"  Questions: {len(items)}")

        # Update status
        await storage.update_question_set_status(qs["id"], "resolving")

        for item in items:
            question = await storage.get_question(item["source"], item["question_id"])
            if not question:
                continue

            # Get forecasts for this question in this question set
            forecasts = await storage.get_forecasts(
                question_id=item["question_id"],
                source=item["source"],
                question_set_id=qs["id"],
            )

            if not forecasts:
                continue

            # For data questions, get value at forecast due date
            resolution_at_due_date = None
            if question.source_type == SourceType.DATA:
                resolution_at_due_date = await storage.get_resolution(
                    source=item["source"],
                    question_id=item["question_id"],
                    resolution_date=forecast_due_date,
                )

            # For each resolution date that has passed
            for res_date in resolution_dates:
                if res_date > today:
                    continue

                # Get resolution value at that date
                resolution = await storage.get_resolution(
                    source=item["source"],
                    question_id=item["question_id"],
                    resolution_date=res_date,
                )

                if resolution is None:
                    # Try to get latest available resolution
                    resolution = await storage.get_resolution(
                        source=item["source"],
                        question_id=item["question_id"],
                    )

                # Use resolution module to handle edge cases
                result = resolve_question(
                    question=question,
                    resolution=resolution,
                    forecast_due_date=forecast_due_date,
                    resolution_at_due_date=resolution_at_due_date,
                )

                if not result.resolved:
                    skipped_count += 1
                    continue

                # Count scorable forecasts (scores computed on-the-fly in leaderboard)
                for fc in forecasts:
                    if fc.id is None:
                        continue

                    brier = score_forecast(fc, result)
                    if brier is not None:
                        total_scores += 1

        # Check if all resolution dates have passed
        if all(rd <= today for rd in resolution_dates):
            await storage.update_question_set_status(qs["id"], "completed")
            click.echo(f"  ✅ Question set completed")

    click.echo(f"\n✅ Scored {total_scores} forecasts ({skipped_count} skipped)")
    await storage.close()


@main.command()
@click.option(
    "--question-set",
    "-q",
    type=int,
    default=None,
    help="Show leaderboard for specific question set. Default: all.",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format.",
)
@click.option(
    "--significance",
    "-s",
    is_flag=True,
    help="Show statistical significance comparisons.",
)
def leaderboard(question_set: int | None, format: str, significance: bool):
    """View current standings."""
    asyncio.run(_leaderboard(question_set, format, significance))


async def _leaderboard(question_set_id: int | None, output_format: str, show_significance: bool):
    """Async implementation of leaderboard command.

    Computes scores on-the-fly from stored forecasts and resolutions.
    """
    from collections import defaultdict
    from datetime import date as date_type

    storage = get_storage()
    today = date_type.today()

    # Get question sets to evaluate
    if question_set_id:
        qs = await storage.get_question_set(question_set_id)
        question_sets = [qs] if qs else []
    else:
        question_sets = await storage.get_question_sets()

    if not question_sets:
        click.echo("No question sets found.")
        await storage.close()
        return

    # Compute scores on-the-fly
    scores_by_forecaster: dict[str, list[float]] = defaultdict(list)
    question_ids: list[str] = []

    for qs in question_sets:
        resolution_dates = [date_type.fromisoformat(d) for d in qs["resolution_dates"]]
        forecast_due_date = qs["forecast_due_date"]
        items = await storage.get_question_set_items(qs["id"])

        for item in items:
            question = await storage.get_question(item["source"], item["question_id"])
            if not question:
                continue

            forecasts = await storage.get_forecasts(
                question_id=item["question_id"],
                source=item["source"],
                question_set_id=qs["id"],
            )

            if not forecasts:
                continue

            # Get resolution data
            resolution_at_due_date = None
            if question.source_type == SourceType.DATA:
                resolution_at_due_date = await storage.get_resolution(
                    source=item["source"],
                    question_id=item["question_id"],
                    resolution_date=forecast_due_date,
                )

            # Use the latest passed resolution date
            for res_date in sorted(resolution_dates, reverse=True):
                if res_date > today:
                    continue

                resolution = await storage.get_resolution(
                    source=item["source"],
                    question_id=item["question_id"],
                    resolution_date=res_date,
                )

                if resolution is None:
                    resolution = await storage.get_resolution(
                        source=item["source"],
                        question_id=item["question_id"],
                    )

                result = resolve_question(
                    question=question,
                    resolution=resolution,
                    forecast_due_date=forecast_due_date,
                    resolution_at_due_date=resolution_at_due_date,
                )

                if not result.resolved:
                    continue

                # Score each forecast
                for fc in forecasts:
                    brier = score_forecast(fc, result)
                    if brier is not None:
                        scores_by_forecaster[fc.forecaster].append(brier)
                        if item["question_id"] not in question_ids:
                            question_ids.append(item["question_id"])

                break  # Only use the latest resolution date

    if not scores_by_forecaster:
        click.echo("No scores yet. Run 'forecast' and 'resolve' first.")
        await storage.close()
        return

    if output_format == "json":
        # Build simple JSON output
        leaderboard = []
        for forecaster, scores in scores_by_forecaster.items():
            leaderboard.append({
                "forecaster": forecaster,
                "mean_brier_score": sum(scores) / len(scores),
                "num_forecasts": len(scores),
            })
        leaderboard.sort(key=lambda x: x["mean_brier_score"])
        click.echo(json.dumps(leaderboard, indent=2))
    else:
        # Build leaderboard with confidence intervals
        entries = build_leaderboard(dict(scores_by_forecaster), with_confidence=True)

        # Compute pairwise significance if requested
        comparisons = None
        if show_significance and len(entries) > 1:
            comparisons = compute_pairwise_significance(
                dict(scores_by_forecaster), question_ids
            )

        # Format and display
        output = format_leaderboard(entries, comparisons)
        click.echo(output)

    await storage.close()


@main.command("question-sets")
@click.option(
    "--status",
    "-s",
    type=click.Choice(["pending", "forecasting", "resolving", "completed"]),
    default=None,
    help="Filter by status.",
)
def list_question_sets(status: str | None):
    """List all question sets."""
    asyncio.run(_list_question_sets(status))


async def _list_question_sets(status: str | None):
    """Async implementation of question-sets command."""
    storage = get_storage()

    sets = await storage.get_question_sets(status=status)

    if not sets:
        click.echo("No question sets found.")
        await storage.close()
        return

    click.echo("\n📋 Question Sets\n")
    click.echo(f"{'ID':<6}{'Name':<25}{'Status':<15}{'Freeze':<12}{'Due':<12}{'Questions':<10}")
    click.echo("-" * 80)

    for qs in sets:
        # Get question count
        items = await storage.get_question_set_items(qs["id"])
        click.echo(
            f"{qs['id']:<6}{qs['name']:<25}{qs['status']:<15}"
            f"{str(qs['freeze_date']):<12}{str(qs['forecast_due_date']):<12}{len(items):<10}"
        )

    await storage.close()


@main.command()
def sources():
    """List available data sources."""
    click.echo("\n📡 Available Sources\n")
    for name in sorted(registry.list()):
        click.echo(f"  • {name}")


@main.command("forecasts")
@click.option(
    "--question-set",
    "-q",
    type=int,
    default=None,
    help="Filter by question set ID.",
)
@click.option(
    "--forecaster",
    "-f",
    default=None,
    help="Filter by forecaster name.",
)
@click.option(
    "--limit",
    "-n",
    type=int,
    default=20,
    help="Maximum number of forecasts to show. Default: 20",
)
def list_forecasts(question_set: int | None, forecaster: str | None, limit: int):
    """List forecasts."""
    asyncio.run(_list_forecasts(question_set, forecaster, limit))


async def _list_forecasts(question_set_id: int | None, forecaster: str | None, limit: int):
    """Async implementation of forecasts command."""
    storage = get_storage()

    forecasts = await storage.get_forecasts(
        question_set_id=question_set_id,
        forecaster=forecaster,
        limit=limit,
    )

    if not forecasts:
        click.echo("No forecasts found.")
        await storage.close()
        return

    click.echo(f"\n🔮 Forecasts (showing {len(forecasts)})\n")
    click.echo(f"{'ID':<8}{'Forecaster':<30}{'Question':<20}{'Prob':<8}{'Created':<12}")
    click.echo("-" * 78)

    for fc in forecasts:
        prob_str = f"{fc.probability:.2f}" if fc.probability is not None else "N/A"
        q_id = fc.question_id[:17] + "..." if len(fc.question_id) > 20 else fc.question_id
        created = fc.created_at.strftime("%Y-%m-%d")
        click.echo(
            f"{fc.id or 0:<8}{fc.forecaster:<30}{q_id:<20}{prob_str:<8}{created:<12}"
        )

    await storage.close()


if __name__ == "__main__":
    main()
