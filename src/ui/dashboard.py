from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, ProgressBar


class GeneWeaverApp(App):
    """GeneWeaver terminal dashboard."""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("GeneWeaver - CRISPR Alignment Engine")
        yield Static("Status: Ready")
        yield Static("Processing Progress:")
        yield ProgressBar(total=100, show_eta=False)
        yield Footer()


if __name__ == "__main__":
    app = GeneWeaverApp()
    app.run()