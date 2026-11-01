"""Textual dashboard launcher and interface for GeneWeaver."""

from src.ui.dashboard import GeneWeaverApp


def run_dashboard() -> None:
    """Launch the GeneWeaver interactive terminal dashboard."""
    app = GeneWeaverApp()
    app.run()


if __name__ == "__main__":
    run_dashboard()
