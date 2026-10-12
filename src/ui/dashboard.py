from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, ProgressBar, Button

from src.alignment.integration import run_pipeline_timed


class GeneWeaverApp(App):
    """GeneWeaver genome alignment dashboard."""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("GeneWeaver - CRISPR Alignment Engine")
        yield Static("Status: Ready", id="status")
        yield Static("Sequences: 0", id="sequences")
        yield Static("Chunks: 0", id="chunks")
        yield Static("Aligned chunks: 0", id="aligned")
        yield Static("Similarity: --", id="similarity")
        yield Static("Pipeline time: --", id="timing")
        yield Static("Alignment Progress:")
        yield ProgressBar(total=100, show_eta=False, id="progress")
        yield Button("Run Alignment", id="start")
        yield Footer()

    def load_genome_data(self) -> None:
        """Load genome data and update sequence and chunk counts."""
        from src.pipeline import process_genome

        sequences, chunks = process_genome(
            "data/mock_genome.fasta",
            10,
        )
        self.query_one("#sequences", Static).update(
            f"Sequences: {len(sequences)}"
        )
        self.query_one("#chunks", Static).update(
            f"Chunks: {len(chunks)}"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "start":
            return

        button = self.query_one("#start", Button)
        progress = self.query_one("#progress", ProgressBar)
        status = self.query_one("#status", Static)
        button.disabled = True
        status.update("Status: Running CPU alignment...")
        progress.update(progress=0)

        try:
            results, timing = run_pipeline_timed(
                fasta_path=Path("data/mock_genome.fasta"),
                target="ATGCGTACGT",
                chunk_size=10,
                backend="cpu",
                skip_unequal=True,
            )

            matches = sum(result.matches for result in results)
            mismatches = sum(result.mismatches for result in results)
            compared = matches + mismatches
            similarity = matches / compared * 100 if compared else 0.0

            self.query_one("#sequences", Static).update(
                f"Sequences: {timing.num_sequences}"
            )
            self.query_one("#chunks", Static).update(
                f"Chunks: {timing.num_chunks}"
            )
            self.query_one("#aligned", Static).update(
                f"Aligned chunks: {timing.num_aligned}"
            )
            self.query_one("#similarity", Static).update(
                f"Similarity: {similarity:.2f}%"
            )
            self.query_one("#timing", Static).update(
                f"Pipeline time: {timing.total_pipeline_time:.2f} ms"
            )
            progress.update(progress=100)
            status.update("Status: Alignment completed")

        except Exception as error:
            progress.update(progress=0)
            status.update(f"Status: Alignment failed: {error}")

        finally:
            button.disabled = False


if __name__ == "__main__":
    GeneWeaverApp().run()
