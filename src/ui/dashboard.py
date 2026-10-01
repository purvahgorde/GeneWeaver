
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, ProgressBar, Button

from src.pipeline import process_genome


class GeneWeaverApp(App):
    """GeneWeaver terminal dashboard."""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("GeneWeaver - CRISPR Alignment Engine")
        yield Static("Status: Ready", id="status")
        yield Static("Sequences: 0", id="sequences")
        yield Static("Chunks: 0", id="chunks")
        yield Static("Demo Progress:")
        yield ProgressBar(total=100, show_eta=False, id="progress")
        yield Button("Start Demo", id="start")
        yield Footer()

    def load_genome_data(self) -> None:
        """Load genome data and update sequence/chunk counts."""
        sequences, chunks = process_genome(
            "data/mock_genome.fasta",
            10,
        )

        self.sequence_count = len(sequences)
        self.chunk_count = len(chunks)

        self.query_one("#sequences", Static).update(
            f"Sequences: {self.sequence_count}"
        )
        self.query_one("#chunks", Static).update(
            f"Chunks: {self.chunk_count}"
        )

    def update_progress(self) -> None:
        """Simulate progress for the dashboard demonstration."""
        progress = self.query_one("#progress", ProgressBar)
        status = self.query_one("#status", Static)

        if progress.progress < 25:
            progress.update(progress=25)
            status.update("Status: Demo started")

        elif progress.progress < 50:
            progress.update(progress=50)
            status.update("Status: Demonstrating progress...")

        elif progress.progress < 75:
            progress.update(progress=75)
            status.update("Status: Almost finished...")

        else:
            progress.update(progress=100)
            status.update("Status: Demo completed")

            self.timer.pause()
            self.query_one("#start", Button).disabled = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the dashboard demo button."""
        if event.button.id != "start":
            return

        start_button = self.query_one("#start", Button)
        progress = self.query_one("#progress", ProgressBar)
        status = self.query_one("#status", Static)

        start_button.disabled = True
        progress.update(progress=0)

        try:
            self.load_genome_data()
        except (OSError, ValueError) as error:
            status.update(f"Status: Could not load genome data: {error}")
            start_button.disabled = False
            return

        self.timer = self.set_interval(1.0, self.update_progress)


if __name__ == "__main__":
    GeneWeaverApp().run()