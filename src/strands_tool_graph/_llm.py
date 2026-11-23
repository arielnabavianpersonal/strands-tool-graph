from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any, TypeVar

from pydantic import BaseModel
from strands.models import Model
from strands.types.content import Messages
from strands.types.event_loop import Usage

OutputT = TypeVar("OutputT", bound=BaseModel)


class UsageMeter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._input_tokens = 0
        self._output_tokens = 0
        self._total_tokens = 0

    @property
    def usage(self) -> Usage:
        with self._lock:
            return Usage(
                inputTokens=self._input_tokens,
                outputTokens=self._output_tokens,
                totalTokens=self._total_tokens,
            )

    def add(self, usage: Mapping[str, Any] | None) -> None:
        if not usage:
            return
        with self._lock:
            self._input_tokens += int(usage.get("inputTokens", 0))
            self._output_tokens += int(usage.get("outputTokens", 0))
            self._total_tokens += int(usage.get("totalTokens", 0))


async def ask(
    model: Model,
    system_prompt: str,
    prompt: str,
    output_type: type[OutputT],
    meter: UsageMeter,
) -> OutputT:
    messages: Messages = [{"role": "user", "content": [{"text": prompt}]}]
    output: Any = None
    usage = StreamUsage()
    async for event in model.structured_output(output_type, messages, system_prompt=system_prompt):
        usage.observe(event)
        output = event.get("output", output)
    meter.add(usage.total)
    if not isinstance(output, output_type):
        raise ValueError(f"model returned no {output_type.__name__}")
    return output


class StreamUsage:
    def __init__(self) -> None:
        self._from_stop: Mapping[str, Any] | None = None
        self._from_metadata: Mapping[str, Any] | None = None

    @property
    def total(self) -> Mapping[str, Any] | None:
        return self._from_stop or self._from_metadata

    def observe(self, event: Mapping[str, Any]) -> None:
        stop = event.get("stop")
        if isinstance(stop, tuple) and len(stop) >= 3 and isinstance(stop[2], Mapping):
            self._from_stop = stop[2]
        chunk = event.get("event")
        metadata = chunk.get("metadata") if isinstance(chunk, Mapping) else None
        if isinstance(metadata, Mapping) and isinstance(metadata.get("usage"), Mapping):
            self._from_metadata = metadata["usage"]
