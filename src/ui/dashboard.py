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

    def update_progress(self) -> None:
        """Update alignment progress step by step."""
        progress = self.query_one("#progress", ProgressBar)
        status = self.query_one("#status", Static)

        if progress.progress < 25:
            progress.update(progress=25)
            status.update("Status: Alignment started")

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

        self.timer = self.set_interval(
            1.0,
            self.update_progress
        )


if __name__ == "__main__":
    app = GeneWeaverApp()
    app.run()