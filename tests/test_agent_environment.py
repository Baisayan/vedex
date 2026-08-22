from __future__ import annotations

from pathlib import Path

from vedex.agent import Agent
from vedex.environments import LocalEnvironment
from vedex.models import (
    FakeAdapter,
    ModelCompletedEvent,
    ModelEvent,
    ModelSettings,
    ModelStartEvent,
)
from vedex.schema import (
    AgentEndEvent,
    AgentEvent,
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
)

from .conftest import run_async


def test_agent_reports_unavailable_environment_as_fatal(tmp_path: Path) -> None:
    environment = LocalEnvironment(tmp_path)
    model_stream: list[ModelEvent] = [
        ModelStartEvent(),
        ModelCompletedEvent(
            message=AssistantMessage(
                tool_calls=[ToolCall(id="call-1", name="read", arguments={"path": "file.txt"})]
            )
        ),
    ]
    agent = Agent(
        adapter=FakeAdapter([model_stream]),
        settings=ModelSettings(model="fake"),
        system_prompt="test",
        tools=environment.tools,
    )

    async def collect() -> list[AgentEvent]:
        return [event async for event in agent.run("Read the file")]

    events = run_async(collect())

    end = next(event for event in events if isinstance(event, AgentEndEvent))
    assert end.status == "fatal_environment_failure"
    assert isinstance(agent.messages[-1], ToolResultMessage)
    assert agent.messages[-1].ok is False
    assert "requires state 'started'" in agent.messages[-1].content
