from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, ProgressBar, Button


class GeneWeaverApp(App):
    """GeneWeaver terminal dashboard."""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("GeneWeaver - CRISPR Alignment Engine")
        yield Static("Status: Ready", id="status")
        yield Static("Processing Progress:")
        yield ProgressBar(total=100, show_eta=False, id="progress")
        yield Button("Start Alignment", id="start")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the Start Alignment button."""
        status = self.query_one("#status", Static)
        progress = self.query_one("#progress", ProgressBar)

        status.update("Status: Alignment started")
        progress.update(progress=25)


if __name__ == "__main__":
    app = GeneWeaverApp()
    app.run()