from __future__ import annotations

import copy
import itertools
import json
from collections.abc import AsyncGenerator, Callable, Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel
from strands.models import Model
from strands.tools.tools import PythonAgentTool
from strands.types.content import Message, Messages
from strands.types.tools import ToolSpec

from strands_tool_graph import EdgeEvidence, ToolEdge, ToolGraph
from strands_tool_graph._builder import node_from_spec

Plan = Callable[[Messages, list[str]], Message]
Answer = Callable[[str], BaseModel]

_ids = itertools.count(1)


def text_message(text: str) -> Message:
    return {"role": "assistant", "content": [{"text": text}]}


def call_message(*calls: tuple[str, Mapping[str, Any]]) -> Message:
    return {
        "role": "assistant",
        "content": [
            {"toolUse": {"toolUseId": f"call-{next(_ids)}", "name": name, "input": dict(arguments)}}
            for name, arguments in calls
        ],
    }


def tool_turn(
    name: str, arguments: Mapping[str, Any], result: Any, status: Literal["success", "error"] = "success"
) -> list[Message]:
    tool_use_id = f"call-{next(_ids)}"
    return [
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": tool_use_id, "name": name, "input": dict(arguments)}}],
        },
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": tool_use_id, "status": status, "content": [{"json": result}]}}],
        },
    ]


class ScriptedModel(Model):
    def __init__(
        self,
        plan: Plan | None = None,
        answers: Mapping[str, Answer] | None = None,
        tokens: int = 10,
    ) -> None:
        self.plan = plan or (lambda messages, tools: text_message("done"))
        self.answers = dict(answers or {})
        self.tokens = tokens
        self.stream_calls: list[dict[str, Any]] = []
        self.structured_calls: list[dict[str, Any]] = []
        self.config: dict[str, Any] = {"model_id": "scripted"}

    def update_config(self, **model_config: Any) -> None:
        self.config.update(model_config)

    def get_config(self) -> Any:
        return self.config

    def calls_for(self, output_type: str) -> list[dict[str, Any]]:
        return [call for call in self.structured_calls if call["type"] == output_type]

    async def structured_output(
        self, output_model: type[BaseModel], prompt: Messages, system_prompt: str | None = None, **kwargs: Any
    ) -> AsyncGenerator[Any, None]:
        text = "\n".join(block.get("text", "") for message in prompt for block in message["content"])
        self.structured_calls.append({"type": output_model.__name__, "prompt": text, "system_prompt": system_prompt})
        answer = self.answers[output_model.__name__](text)
        yield {"event": {"metadata": {"usage": self._usage(), "metrics": {"latencyMs": 0}}}}
        yield {"output": answer}

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[Any] | None = None,
        system_prompt: str | None = None,
        tool_choice: Any | None = None,
        *,
        system_prompt_content: Any = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        tools = [spec["name"] for spec in tool_specs or []]
        self.stream_calls.append({"messages": copy.deepcopy(messages), "tools": tools, "system_prompt": system_prompt})
        for event in _events(self.plan(messages, tools), self._usage()):
            yield event

    def _usage(self) -> dict[str, int]:
        return {"inputTokens": self.tokens, "outputTokens": 1, "totalTokens": self.tokens + 1}


def _events(message: Message, usage: Mapping[str, int]) -> Iterable[dict[str, Any]]:
    stop_reason = "end_turn"
    yield {"messageStart": {"role": "assistant"}}
    for block in message["content"]:
        if "text" in block:
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": block["text"]}}}
            yield {"contentBlockStop": {}}
        if "toolUse" in block:
            stop_reason = "tool_use"
            tool_use = block["toolUse"]
            yield {
                "contentBlockStart": {
                    "start": {"toolUse": {"name": tool_use["name"], "toolUseId": tool_use["toolUseId"]}}
                }
            }
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(tool_use["input"])}}}}
            yield {"contentBlockStop": {}}
    yield {"messageStop": {"stopReason": stop_reason}}
    yield {"metadata": {"usage": dict(usage), "metrics": {"latencyMs": 0}}}


def make_tool(
    name: str,
    description: str,
    inputs: Mapping[str, str],
    *,
    optional: Iterable[str] = (),
    outputs: Mapping[str, str] | None = None,
    result: Mapping[str, Any] | None = None,
) -> PythonAgentTool:
    spec: ToolSpec = {
        "name": name,
        "description": description,
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {field: {"type": kind, "description": f"The {field}."} for field, kind in inputs.items()},
                "required": [field for field in inputs if field not in set(optional)],
            }
        },
    }
    if outputs is not None:
        spec["outputSchema"] = {
            "json": {"type": "object", "properties": {field: {"type": kind} for field, kind in outputs.items()}}
        }

    def run(tool_use: Any, **_: Any) -> Any:
        return {"toolUseId": tool_use["toolUseId"], "status": "success", "content": [{"json": dict(result or {})}]}

    run.__name__ = name
    return PythonAgentTool(name, spec, run)


def shop_tools() -> list[PythonAgentTool]:
    return [
        make_tool("find_user", "Find a customer by email.", {"email": "string"}, result={"user_id": "u-1"}),
        make_tool("list_orders", "List the orders of a customer.", {"user_id": "string"}, result={"order_id": "o-7"}),
        make_tool(
            "refund_order",
            "Refund an order.",
            {"order_id": "string", "reason": "string"},
            result={"refunded": True},
        ),
        make_tool("get_weather", "Get the weather forecast for a city.", {"city": "string"}, result={"sky": "sun"}),
    ]


def edge(source: str, target: str, supplies: str, weight: float) -> ToolEdge:
    return ToolEdge(
        source=source,
        target=target,
        supplies=(supplies,),
        weight=weight,
        evidence=(EdgeEvidence("model", weight, {"input": supplies}),),
    )


def shop_graph(find_to_list: float = 0.9, list_to_refund: float = 0.8) -> ToolGraph:
    nodes = {tool.tool_name: node_from_spec(tool.tool_spec) for tool in shop_tools()}
    return ToolGraph(
        nodes=nodes,
        edges=(
            edge("find_user", "list_orders", "user_id", find_to_list),
            edge("list_orders", "refund_order", "order_id", list_to_refund),
        ),
    )
