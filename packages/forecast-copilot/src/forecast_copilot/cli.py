"""Forecast Copilot CLI.

Interactive command-line interface for forecasting assistance.

TODO: Implement after modes are functional.
"""

import click


@click.group()
def main():
    """Forecast Copilot - Interactive forecasting assistant."""
    pass


@main.command()
@click.argument("question")
@click.option("--mode", type=click.Choice(["forecaster", "sounding_board"]), default="sounding_board")
def ask(question: str, mode: str):
    """Ask the copilot about a forecasting question."""
    click.echo(f"Mode: {mode}")
    click.echo(f"Question: {question}")
    click.echo("TODO: Implement copilot interaction")


@main.command()
def interactive():
    """Start an interactive forecasting session."""
    click.echo("TODO: Implement interactive mode")


if __name__ == "__main__":
    main()
