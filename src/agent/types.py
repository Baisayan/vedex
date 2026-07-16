from typing import Protocol


class CancellationToken(Protocol):
    def is_cancelled(self) -> bool: ...


type JSONPrimitive = str | int | float | bool | None
type JSONValue = JSONPrimitive | list[JSONValue] | dict[str, JSONValue]
type JSONObject = dict[str, JSONValue]