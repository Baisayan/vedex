from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from pydantic import TypeAdapter, ValidationError
from vedex.models import (
    FakeAdapter,
    ModelAdapter,
    ModelCancelledEvent,
    ModelCompletedEvent,
    ModelEvent,
    ModelFailureEvent,
    ModelRequest,
    ModelSettings,
    ModelStartEvent,
    ModelTextDeltaEvent,
    ModelThinkingDeltaEvent,
    ModelToolDefinition,
    Usage,
)
from vedex.schema import AssistantMessage, CancellationToken, ToolCall, UserMessage

from .conftest import make_tool, run_async


async def _collect(
    adapter: ModelAdapter,
    request: ModelRequest,
    *,
    signal: CancellationToken | None = None,
) -> list[ModelEvent]:
    return [event async for event in adapter.stream(request, signal=signal)]


def _request() -> ModelRequest:
    return ModelRequest.from_agent_inputs(
        system="You are a coding agent.",
        messages=[UserMessage(content="Read pyproject.toml")],
        tools=[make_tool(name="read")],
        settings=ModelSettings(
            model="fake-model",
            options={"temperature": 0, "nested": {"enabled": True}},
        ),
    )


def test_model_contracts_round_trip_with_usage_tools_and_opaque_metadata() -> None:
    event_adapter: TypeAdapter[ModelEvent] = TypeAdapter(ModelEvent)
    events: list[ModelEvent] = [
        ModelStartEvent(),
        ModelTextDeltaEvent(delta="I will inspect it."),
        ModelThinkingDeltaEvent(delta="Need the read tool."),
        ModelCompletedEvent(
            message=AssistantMessage(
                content="I will inspect it.",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="read",
                        arguments={"path": "pyproject.toml"},
                    )
                ],
                metadata={"continuation": {"id": "response-1"}, "cache_hit": False},
            ),
            usage=Usage(
                input_tokens=120,
                output_tokens=18,
                cached_tokens=40,
                thinking_tokens=7,
            ),
        ),
        ModelFailureEvent(kind="rate_limit", message="Try later", retryable=True),
        ModelCancelledEvent(),
    ]

    assert [
        event_adapter.validate_json(event_adapter.dump_json(event)) for event in events
    ] == events


@pytest.mark.parametrize(
    ("model", "value"),
    [
        (AssistantMessage, {"metadata": {"not_json": object()}}),
        (Usage, {"input_tokens": -1}),
        (ModelSettings, {"model": "", "options": {}}),
        (
            ModelToolDefinition,
            {
                "name": "read",
                "description": "Read",
                "input_schema": {"x": object()},
            },
        ),
    ],
)
def test_model_contracts_reject_invalid_boundary_data(
    model: type[AssistantMessage] | type[Usage] | type[ModelSettings] | type[ModelToolDefinition],
    value: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(value)


def test_model_request_contains_only_normalized_tool_definition() -> None:
    request = _request()

    assert request.system == "You are a coding agent."
    assert request.messages == [UserMessage(content="Read pyproject.toml")]
    assert request.tools == [
        ModelToolDefinition(
            name="read",
            description="read description",
            input_schema={"type": "object"},
        )
    ]
    assert request.settings.options == {
        "temperature": 0,
        "nested": {"enabled": True},
    }
    assert not hasattr(request.tools[0], "executor")


def test_fake_adapter_replays_stream_and_records_request_snapshot() -> None:
    completed = ModelCompletedEvent(
        message=AssistantMessage(
            content="Reading now.",
            tool_calls=[ToolCall(id="call-1", name="read", arguments={"path": "README.md"})],
            metadata={"response_id": "fake-1"},
        ),
        usage=Usage(input_tokens=10, output_tokens=4, cached_tokens=2),
    )
    fake = FakeAdapter([[ModelStartEvent(), ModelTextDeltaEvent(delta="Reading now."), completed]])
    adapter: ModelAdapter = fake
    request = _request()

    events = run_async(_collect(adapter, request))
    request.messages.append(UserMessage(content="A later mutation"))

    assert events == [ModelStartEvent(), ModelTextDeltaEvent(delta="Reading now."), completed]
    assert fake.requests[0].messages == [UserMessage(content="Read pyproject.toml")]
    assert fake.remaining_streams == 0


def test_fake_adapter_replays_failure_and_normalizes_exhaustion() -> None:
    failure = ModelFailureEvent(
        kind="unavailable",
        message="Synthetic outage",
        retryable=True,
    )
    fake = FakeAdapter([[failure]])
    request = _request()

    assert run_async(_collect(fake, request)) == [failure]
    assert run_async(_collect(fake, request)) == [
        ModelFailureEvent(message="FakeAdapter has no scripted response")
    ]
    assert len(fake.requests) == 2


def test_fake_adapter_turns_signal_cancellation_into_terminal_event() -> None:
    token = _MutableCancellationToken()
    fake = FakeAdapter(
        [
            [
                ModelStartEvent(),
                ModelTextDeltaEvent(delta="must not be yielded"),
                ModelCompletedEvent(message=AssistantMessage(content="done")),
            ]
        ]
    )

    async def cancel_after_start() -> list[ModelEvent]:
        stream: AsyncIterator[ModelEvent] = fake.stream(_request(), signal=token)
        events = [await anext(stream)]
        token.cancel()
        events.extend([event async for event in stream])
        return events

    assert run_async(cancel_after_start()) == [ModelStartEvent(), ModelCancelledEvent()]


class _MutableCancellationToken:
    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled
