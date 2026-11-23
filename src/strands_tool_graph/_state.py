from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from strands.types.content import ContentBlock, Message, Messages

_MAX_BLOCK_CHARS = 2_000
_MAX_TRANSCRIPT_CHARS = 24_000
_MAX_ARGUMENT_CHARS = 200
_SHORT_VALUE_CHARS = 3


@dataclass(frozen=True)
class ToolCall:
    name: str
    input: Mapping[str, Any]

    def describe(self) -> str:
        arguments = ", ".join(
            f"{key}={_clip(json.dumps(value, default=str), _MAX_ARGUMENT_CHARS)}" for key, value in self.input.items()
        )
        return f"{self.name}({arguments})"


@dataclass(frozen=True)
class Candidate:
    calls: tuple[ToolCall, ...]
    text: str = ""

    @classmethod
    def from_message(cls, message: Message) -> Candidate:
        calls = tuple(
            ToolCall(name=block["toolUse"]["name"], input=dict(block["toolUse"].get("input") or {}))
            for block in message["content"]
            if "toolUse" in block
        )
        text = " ".join(block["text"] for block in message["content"] if "text" in block).strip()
        return cls(calls=calls, text=text)

    @property
    def tools(self) -> tuple[str, ...]:
        return tuple(call.name for call in self.calls)

    @property
    def key(self) -> str:
        if not self.calls:
            return "reply"
        return json.dumps([[call.name, call.input] for call in self.calls], sort_keys=True, default=str)

    def describe(self) -> str:
        if not self.calls:
            return f"reply to the user: {_clip(self.text, _MAX_ARGUMENT_CHARS)}" if self.text else "reply to the user"
        return " then ".join(call.describe() for call in self.calls)


@dataclass(frozen=True)
class DecisionState:
    request: str
    transcript: str
    grounding: str
    successes: Mapping[str, int]
    failures: Mapping[str, int]

    @classmethod
    def from_messages(cls, messages: Messages) -> DecisionState:
        statuses = _tool_statuses(messages)
        return cls(
            request="\n".join(_user_asks(messages)),
            transcript=_transcript(messages),
            grounding=" ".join(_texts(messages)).lower(),
            successes=Counter(name for name, status in statuses if status == "success"),
            failures=Counter(name for name, status in statuses if status != "success"),
        )

    @property
    def succeeded(self) -> frozenset[str]:
        return frozenset(name for name, count in self.successes.items() if count)

    @property
    def failed_only(self) -> frozenset[str]:
        return frozenset(name for name, count in self.failures.items() if count and not self.successes.get(name))

    def grounds(self, value: Any) -> bool:
        text = (value if isinstance(value, str) else json.dumps(value, default=str)).strip().lower()
        if not text:
            return False
        if len(text) < _SHORT_VALUE_CHARS:
            return re.search(rf"(?<![a-z0-9]){re.escape(text)}(?![a-z0-9])", self.grounding) is not None
        return text in self.grounding

    def history(self, name: str) -> str:
        successes, failures = self.successes.get(name, 0), self.failures.get(name, 0)
        if not successes and not failures:
            return "not called yet"
        return f"called {successes + failures}x ({successes} ok, {failures} failed)"


def _user_asks(messages: Messages) -> list[str]:
    return [
        block["text"]
        for message in messages
        if message["role"] == "user" and not any("toolResult" in block for block in message["content"])
        for block in message["content"]
        if "text" in block
    ]


def _tool_statuses(messages: Messages) -> list[tuple[str, str]]:
    names = {
        block["toolUse"]["toolUseId"]: block["toolUse"]["name"]
        for message in messages
        for block in message["content"]
        if "toolUse" in block
    }
    return [
        (names[block["toolResult"]["toolUseId"]], block["toolResult"].get("status", "success"))
        for message in messages
        for block in message["content"]
        if "toolResult" in block and block["toolResult"]["toolUseId"] in names
    ]


def _texts(messages: Messages) -> list[str]:
    return [text for message in messages for block in message["content"] for text in _block_texts(block)]


def _block_texts(block: ContentBlock) -> list[str]:
    if "text" in block:
        return [block["text"]]
    if "toolResult" in block:
        return [_result_text(block["toolResult"].get("content", []))]
    return []


def _result_text(content: Any) -> str:
    parts = [
        part.get("text") if "text" in part else json.dumps(part.get("json"), default=str)
        for part in content
        if "text" in part or "json" in part
    ]
    return " ".join(part for part in parts if part)


def _transcript(messages: Messages) -> str:
    lines = [
        line for message in messages for block in message["content"] if (line := _render_block(message["role"], block))
    ]
    return "\n".join(lines)[-_MAX_TRANSCRIPT_CHARS:]


def _render_block(role: str, block: ContentBlock) -> str | None:
    if "text" in block:
        return f"{role}: {_clip(block['text'], _MAX_BLOCK_CHARS)}"
    if "toolUse" in block:
        call = ToolCall(name=block["toolUse"]["name"], input=block["toolUse"].get("input") or {})
        return f"{role} called {call.describe()}"
    if "toolResult" in block:
        result = block["toolResult"]
        text = _clip(_result_text(result.get("content", [])), _MAX_BLOCK_CHARS)
        return f"tool result [{result.get('status', 'success')}]: {text}"
    return None


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[: limit - 3]}..."
