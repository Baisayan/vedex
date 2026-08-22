from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Iterable

from ..schema import CancellationToken
from .base import (
    ModelCancelledEvent,
    ModelEvent,
    ModelFailureEvent,
    ModelRequest,
)


class FakeAdapter:
    """Deterministic adapter that replays scripted normalized event streams."""

    def __init__(self, streams: Iterable[Iterable[ModelEvent]] = ()) -> None:
        self._streams = deque(tuple(stream) for stream in streams)
        self.requests: list[ModelRequest] = []

    @property
    def remaining_streams(self) -> int:
        return len(self._streams)

    def stream(
        self,
        request: ModelRequest,
        *,
        signal: CancellationToken | None = None,
    ) -> AsyncIterator[ModelEvent]:
        self.requests.append(request.model_copy(deep=True))
        scripted = self._streams.popleft() if self._streams else None

        async def replay() -> AsyncIterator[ModelEvent]:
            if signal is not None and signal.is_cancelled():
                yield ModelCancelledEvent()
                return

            if scripted is None:
                yield ModelFailureEvent(message="FakeAdapter has no scripted response")
                return

            for event in scripted:
                if signal is not None and signal.is_cancelled():
                    yield ModelCancelledEvent()
                    return
                yield event

        return replay()


__all__ = ["FakeAdapter"]
