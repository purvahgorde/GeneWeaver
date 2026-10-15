
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, ProgressBar, Button

from src.alignment.integration import run_pipeline_timed
from src.parser.fasta_parser import parse_fasta
from src.chunking.genome_chunker import chunk_sequence


class GeneWeaverApp(App):
    """GeneWeaver genome alignment dashboard."""

    TARGET = "ATGCGTACGT"
    CHUNK_SIZE = 10
    FASTA_PATH = Path("data/mock_genome.fasta")

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("GeneWeaver - CRISPR Alignment Engine")
        yield Static("Status: Ready", id="status")
        yield Static("Sequences: 0", id="sequences")
        yield Static("Chunks: 0", id="chunks")
        yield Static("Aligned chunks: 0", id="aligned")
        yield Static("Matches: 0", id="matches")
        yield Static("Mismatches: 0", id="mismatches")
        yield Static("Similarity: --", id="similarity")
        yield Static("Average alignment score: --", id="score")
        yield Static("Mismatch positions: --", id="positions")
        yield Static("DNA Alignment Details:", id="dna_heading")
        yield Static(
            "Run alignment to compare DNA bases.",
            id="dna_details",
        )
        yield Static("Pipeline time: --", id="timing")
        yield Static("Alignment Progress:")
        yield ProgressBar(total=100, show_eta=False, id="progress")
        yield Button("Run Alignment", id="start")
        yield Footer()

    def load_genome_data(self) -> None:
        """Load genome data and update sequence and chunk counts."""
        from src.pipeline import process_genome

        sequences, chunks = process_genome(
            str(self.FASTA_PATH),
            self.CHUNK_SIZE,
        )

        self.query_one("#sequences", Static).update(
            f"Sequences: {len(sequences)}"
        )
        self.query_one("#chunks", Static).update(
            f"Chunks: {len(chunks)}"
        )

    def format_dna_alignment(self, chunks, target: str) -> Text:
        """Color DNA bases green for matches and red for mismatches."""
        output = Text()

        output.append(f"Target: {target}\n", style="bold cyan")
        output.append(
            "Green = match | Red = mismatch\n\n",
            style="dim",
        )

        if not chunks:
            output.append("No chunks available for comparison.")
            return output

        for index, chunk in enumerate(chunks, start=1):
            output.append(f"Chunk {index}: ", style="bold")

            for base, target_base in zip(chunk, target):
                if base == target_base:
                    output.append(base, style="green")
                else:
                    output.append(base, style="red")

            output.append("\n")

        return output

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Run alignment and update the dashboard."""
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
                fasta_path=self.FASTA_PATH,
                target=self.TARGET,
                chunk_size=self.CHUNK_SIZE,
                backend="cpu",
                skip_unequal=True,
            )

            # Load the original chunks to display the actual DNA bases.
            sequences = parse_fasta(self.FASTA_PATH)

            all_chunks = [
                chunk
                for sequence in sequences
                for chunk in chunk_sequence(sequence, self.CHUNK_SIZE)
            ]

            # The pipeline skips chunks whose lengths differ from the target.
            aligned_chunks = [
                chunk
                for chunk in all_chunks
                if len(chunk) == len(self.TARGET)
            ]

            self.query_one("#dna_details", Static).update(
                self.format_dna_alignment(aligned_chunks, self.TARGET)
            )

            # Calculate alignment statistics.
            matches = sum(result.matches for result in results)
            mismatches = sum(result.mismatches for result in results)
            compared = matches + mismatches

            similarity = (
                matches / compared * 100
                if compared
                else 0.0
            )

            average_score = (
                sum(result.score for result in results) / len(results)
                if results
                else 0.0
            )

            mismatch_positions = [
                f"Chunk {index + 1}: "
                + ", ".join(
                    str(position + 1)
                    for position in result.mismatch_positions
                )
                for index, result in enumerate(results)
                if result.mismatch_positions
            ]

            # Update dashboard metrics.
            self.query_one("#sequences", Static).update(
                f"Sequences: {timing.num_sequences}"
            )
            self.query_one("#chunks", Static).update(
                f"Chunks: {timing.num_chunks}"
            )
            self.query_one("#aligned", Static).update(
                f"Aligned chunks: {timing.num_aligned}"
            )
            self.query_one("#matches", Static).update(
                f"Matches: {matches}"
            )
            self.query_one("#mismatches", Static).update(
                f"Mismatches: {mismatches}"
            )
            self.query_one("#similarity", Static).update(
                f"Similarity: {similarity:.2f}%"
            )
            self.query_one("#score", Static).update(
                f"Average alignment score: {average_score:.4f}"
            )
            self.query_one("#positions", Static).update(
                "Mismatch positions (1-based): "
                + (
                    "; ".join(mismatch_positions)
                    if mismatch_positions
                    else "None"
                )
            )
            self.query_one("#timing", Static).update(
                "Pipeline time: "
                f"{timing.total_pipeline_time * 1000:.2f} ms"
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
