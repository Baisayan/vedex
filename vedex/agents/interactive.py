import sys
from typing import NoReturn

from prompt_toolkit.formatted_text.html import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import PromptSession
from rich.console import Console
from rich.rule import Rule

from vedex import global_config_dir
from vedex.agents.agent import DefaultAgent
from vedex.exceptions import (
    LimitsExceeded,
    TimeExceeded,
    UserInterruption,
)
from vedex.models.utils.content_string import get_content_string

_history = FileHistory(global_config_dir / "interactive_history.txt")
prompt_session = PromptSession(history=_history)
_multiline_prompt_session = PromptSession(history=_history, multiline=True)
console = Console(highlight=False)


def _multiline_prompt() -> str:
    return _multiline_prompt_session.prompt(
        "",
        bottom_toolbar=HTML(
            "Submit message: <b fg='yellow' bg='black'>Esc, then Enter</b> | "
            "Navigate history: <b fg='yellow' bg='black'>Arrow Up/Down</b> | "
            "Search history: <b fg='yellow' bg='black'>Ctrl+R</b>"
        ),
    )


class InteractiveAgent(DefaultAgent):
    def _interrupt(self, content: str, *, itype: str = "UserInterruption") -> NoReturn:
        raise UserInterruption(
            {"role": "user", "content": content, "extra": {"interrupt_type": itype}}
        )

    def add_messages(self, *messages: dict) -> list[dict]:
        for msg in messages:
            role, content = msg.get("role") or msg.get("type", "unknown"), get_content_string(msg)
            if role == "assistant":
                console.print(
                    f"\n[red][bold]vedex[/bold] (step [bold]{self.n_calls}[/bold], "
                    f"[bold]${self.cost:.2f}[/bold]):[/red]\n",
                    end="",
                    highlight=False,
                )
            else:
                console.print(
                    f"\n[bold green]{role.capitalize()}[/bold green]:\n", end="", highlight=False
                )
            console.print(content, highlight=False, markup=False)
        return super().add_messages(*messages)

    def query(self) -> dict:
        try:
            with console.status("Waiting for the LM to respond..."):
                return super().query()
        except TimeExceeded:
            raise
        except LimitsExceeded:
            if not self._stdin_is_interactive():
                raise
            console.print(
                f"Limits exceeded. Limits: {self.config.step_limit} steps, "
                f"${self.config.cost_limit}.\n"
                f"Current spend: {self.n_calls} steps, ${self.cost:.2f}."
            )
            self.config.step_limit = int(input("New step limit: "))
            self.config.cost_limit = float(input("New cost limit: "))
            return super().query()

    @staticmethod
    def _stdin_is_interactive() -> bool:
        try:
            return sys.stdin is not None and sys.stdin.isatty()
        except (ValueError, OSError):
            return False

    def step(self) -> list[dict]:
        try:
            console.print(Rule())
            return super().step()
        except KeyboardInterrupt:
            console.print(
                "\n\n[bold yellow]Interrupted.[/bold yellow] "
                "[green]Add a comment or press Enter to stop.[/green]\n"
                "[bold yellow]>[/bold yellow] ",
                end="",
            )
            interruption_message = (
                prompt_session.prompt("").strip() or "Temporary interruption caught."
            )
            self._interrupt(f"Interrupted by user: {interruption_message}")

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        outputs = []
        try:
            for action in actions:
                outputs.append(self.env.execute(action))
        finally:
            result = self.add_messages(
                *self.model.format_observation_messages(message, outputs, self.get_template_vars())
            )
        return result
