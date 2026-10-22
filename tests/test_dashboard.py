"""Tests for the GeneWeaver Textual dashboard."""

import pytest
from textual.widgets import Static

from src.ui.dashboard import GeneWeaverApp


@pytest.mark.anyio
async def test_dashboard_loads_genome_data():
    """Dashboard should load and display genome statistics."""
    app = GeneWeaverApp()

    async with app.run_test():
        app.load_genome_data()

        sequences_widget = app.query_one("#sequences", Static)
        chunks_widget = app.query_one("#chunks", Static)

        assert str(sequences_widget.content) == "Sequences: 2"
        assert str(chunks_widget.content) == "Chunks: 10"