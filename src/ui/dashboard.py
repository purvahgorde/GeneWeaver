from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, ProgressBar, Button

from src.pipeline import process_genome
from src.parser.fasta_parser import parse_fasta


class GeneWeaverApp(App):
    """GeneWeaver terminal dashboard."""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("GeneWeaver - CRISPR Alignment Engine")
        yield Static("Status: Ready", id="status")
        yield Static("Sequences: 0", id="sequences")
        yield Static("Chunks: 0", id="chunks")
        yield Static("Processing Progress:")
        yield ProgressBar(total=100, show_eta=False, id="progress")
        yield Button("Start Alignment", id="start")
        yield Footer()

    def load_genome_data(self) -> None:
        """Load genome data and update sequence/chunk counts."""
        file_path = "data/mock_genome.fasta"

        sequences = parse_fasta(file_path)
        chunks = process_genome(file_path, 10)

        self.sequence_count = len(sequences)
        self.chunk_count = len(chunks)

        sequences_widget = self.query_one("#sequences", Static)
        chunks_widget = self.query_one("#chunks", Static)

        sequences_widget.update(
            f"Sequences: {self.sequence_count}"
        )

        chunks_widget.update(
            f"Chunks: {self.chunk_count}"
        )

    def update_progress(self) -> None:
        """Update alignment progress step by step."""
        progress = self.query_one("#progress", ProgressBar)
        status = self.query_one("#status", Static)

        if progress.progress < 25:
            progress.update(progress=25)

            status.update(
                f"Status: Alignment started | "
                f"Sequences: {self.sequence_count} | "
                f"Chunks: {self.chunk_count}"
            )

        elif progress.progress < 50:
            progress.update(progress=50)
            status.update("Status: Processing...")

        elif progress.progress < 75:
            progress.update(progress=75)
            status.update("Status: Finalizing...")

        else:
            progress.update(progress=100)
            status.update("Status: Alignment completed")

            self.timer.pause()

            start_button = self.query_one("#start", Button)
            start_button.disabled = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the Start Alignment button."""
        progress = self.query_one("#progress", ProgressBar)
        start_button = self.query_one("#start", Button)

        start_button.disabled = True
        progress.update(progress=0)

        self.load_genome_data()

        self.timer = self.set_interval(
            1.0,
            self.update_progress
        )


if __name__ == "__main__":
    app = GeneWeaverApp()
    app.run()