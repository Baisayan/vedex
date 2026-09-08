import logging
import time
from typing import Any

from pydantic import BaseModel

from vedex.models import GLOBAL_MODEL_STATS
from vedex.models.toolcall import format_toolcall_observation_messages


def make_output(content: str | None, tool_calls: list[dict], actions: list[dict]) -> dict:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
        "extra": {"actions": actions, "cost": 1.0, "timestamp": time.time()},
    }


def _process_test_actions(actions: list[dict]) -> bool:
    for action in actions:
        if "raise" in action:
            raise action["raise"]
        cmd = action.get("command", "")
        if cmd.startswith("/sleep "):
            time.sleep(float(cmd.split("/sleep ")[1]))
            return True
        if cmd.startswith("/warning"):
            logging.warning(cmd.split("/warning")[1])
            return True
    return False


class FakeModelConfig(BaseModel):
    outputs: list[dict]
    model_name: str = "fake"
    cost_per_call: float = 1.0
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n"
        "{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )


class FakeModel:
    def __init__(self, **kwargs):
        self.config = FakeModelConfig(**kwargs)
        self.current_index = -1

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        self.current_index += 1
        output = self.config.outputs[self.current_index]
        if _process_test_actions(output.get("extra", {}).get("actions", [])):
            return self.query(messages, **kwargs)
        GLOBAL_MODEL_STATS.add(self.config.cost_per_call)
        return output

    def format_message(self, **kwargs) -> dict:
        return kwargs

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }
