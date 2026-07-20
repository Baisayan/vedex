from time import time
from typing import Annotated, Literal
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field

from agent.messages import AgentMessage
from agent.types import JSONValue


def new_entry_id() -> str:
    return uuid4().hex


def current_timestamp() -> float:
    return time()


class BaseSessionEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_entry_id)
    timestamp: float = Field(default_factory=current_timestamp)


class MessageEntry(BaseSessionEntry):
    type: Literal["message"] = "message"
    message: AgentMessage


class ModelChangeEntry(BaseSessionEntry):
    type: Literal["model_change"] = "model_change"
    model: str


class CompactionEntry(BaseSessionEntry):
    type: Literal["compaction"] = "compaction"
    summary: str
    replaces_entry_ids: list[str] = Field(default_factory=list)


class SessionInfoEntry(BaseSessionEntry):
    type: Literal["session_info"] = "session_info"
    created_at: float = Field(default_factory=current_timestamp)
    cwd: str | None = None
    title: str | None = None


type SessionEntry = Annotated[
    MessageEntry
    | ModelChangeEntry
    | CompactionEntry
    | SessionInfoEntry,
    Field(discriminator="type"),
]
