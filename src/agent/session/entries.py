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
    parent_id: str | None = None
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


class BranchSummaryEntry(BaseSessionEntry):
    type: Literal["branch_summary"] = "branch_summary"
    summary: str
    branch_root_id: str | None = None


class LabelEntry(BaseSessionEntry):
    type: Literal["label"] = "label"
    label: str


class LeafEntry(BaseSessionEntry):
    type: Literal["leaf"] = "leaf"
    entry_id: str | None = None


class SessionInfoEntry(BaseSessionEntry):
    type: Literal["session_info"] = "session_info"
    created_at: float = Field(default_factory=current_timestamp)
    cwd: str | None = None
    title: str | None = None


class CustomEntry(BaseSessionEntry):
    type: Literal["custom"] = "custom"
    namespace: str
    data: dict[str, JSONValue] = Field(default_factory=dict)


type SessionEntry = Annotated[
    MessageEntry
    | ModelChangeEntry
    | CompactionEntry
    | BranchSummaryEntry
    | LabelEntry
    | LeafEntry
    | SessionInfoEntry
    | CustomEntry,
    Field(discriminator="type"),
]