from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, TypeVar

from strands import Agent
from strands.models import Model
from strands.types.tools import AgentTool, ToolSpec

from ._llm import UsageMeter, ask
from ._prompts import (
    DEPENDENCY_SYSTEM_PROMPT,
    PROBE_JUDGE_SYSTEM_PROMPT,
    PROBE_SYSTEM_PROMPT,
    ProbeVerdict,
    ToolDependencies,
    ToolDependency,
    dependency_prompt,
    probe_judge_prompt,
)
from ._state import DecisionState
from .graph import EdgeEvidence, ToolEdge, ToolField, ToolGraph, ToolNode

logger = logging.getLogger(__name__)

_MODEL_CONCURRENCY = 8
_PROBE_CONCURRENCY = 4
_PROBE_TURNS = 6
_SCHEMA_CONFIDENCE = 1.0
_NAME_NOISE = re.compile(r"[^a-z0-9]")

ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class _Proposal:
    source: str
    target: str
    input: str
    evidence: EdgeEvidence
    probe_query: str = ""


async def build_graph(
    tools: Sequence[AgentTool],
    *,
    model: Model | None,
    probe: bool,
    min_edge_weight: float,
    base: ToolGraph | None,
) -> ToolGraph:
    if not 0 <= min_edge_weight <= 1:
        raise ValueError(f"min_edge_weight must be between 0 and 1, got {min_edge_weight}")
    by_name = _unique_tools(tools)
    builder_model = model if model is not None else _default_model()
    meter = UsageMeter()
    nodes = {name: node_from_spec(tool.tool_spec) for name, tool in by_name.items()}
    known = _unchanged(nodes, base)
    fresh = [name for name in nodes if name not in known]
    proposals = [
        *_schema_proposals(nodes, fresh),
        *await _model_proposals(nodes, fresh, builder_model, meter),
    ]
    if probe:
        proposals = await _probed(proposals, by_name, builder_model, meter)
    kept = [
        edge
        for edge in (base.edges if base else ())
        if edge.source in known and edge.target in known and edge.weight > min_edge_weight
    ]
    edges = sorted([*kept, *_merge(proposals, min_edge_weight)], key=lambda edge: (edge.source, edge.target))
    logger.info(
        "tools=<%d>, analyzed=<%d>, edges=<%d>, total_tokens=<%d> | built tool graph",
        len(nodes),
        len(fresh),
        len(edges),
        meter.usage["totalTokens"],
    )
    return ToolGraph(nodes=nodes, edges=tuple(edges))


def node_from_spec(spec: ToolSpec) -> ToolNode:
    input_schema = _unwrap(spec.get("inputSchema"))
    output_schema = _unwrap(spec.get("outputSchema"))
    canonical = json.dumps(
        [spec["name"], spec.get("description", ""), input_schema, output_schema], sort_keys=True, default=str
    )
    return ToolNode(
        name=spec["name"],
        description=spec.get("description", ""),
        inputs=_fields(input_schema),
        outputs=_fields(output_schema),
        fingerprint=hashlib.sha256(canonical.encode()).hexdigest()[:16],
    )


def _default_model() -> Model:
    from strands.models import BedrockModel

    return BedrockModel()


def _unique_tools(tools: Sequence[AgentTool]) -> dict[str, AgentTool]:
    by_name: dict[str, AgentTool] = {}
    for tool in tools:
        if tool.tool_name in by_name:
            raise ValueError(f"tool names must be unique, found {tool.tool_name!r} twice")
        by_name[tool.tool_name] = tool
    return by_name


def _unwrap(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, Mapping):
        return {}
    inner = schema.get("json", schema)
    return dict(inner) if isinstance(inner, Mapping) else {}


def _fields(schema: Mapping[str, Any]) -> tuple[ToolField, ...]:
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return ()
    required = set(schema.get("required") or ())
    return tuple(
        ToolField(
            name=name,
            type=_type_name(definition),
            description=str(definition.get("description", "")) if isinstance(definition, Mapping) else "",
            required=name in required,
        )
        for name, definition in properties.items()
    )


def _type_name(definition: Any) -> str:
    if not isinstance(definition, Mapping):
        return ""
    declared = definition.get("type")
    if isinstance(declared, str):
        return declared
    if isinstance(declared, list):
        return "|".join(str(item) for item in declared if item != "null")
    variants = definition.get("anyOf") or definition.get("oneOf") or []
    return "|".join(name for variant in variants if (name := _type_name(variant)) and name != "null")


def _unchanged(nodes: Mapping[str, ToolNode], base: ToolGraph | None) -> frozenset[str]:
    if base is None:
        return frozenset()
    return frozenset(
        name for name, node in nodes.items() if name in base.nodes and base.nodes[name].fingerprint == node.fingerprint
    )


def _schema_proposals(nodes: Mapping[str, ToolNode], fresh: Iterable[str]) -> list[_Proposal]:
    focus = set(fresh)
    return [
        _Proposal(
            source=source.name,
            target=target.name,
            input=required.name,
            evidence=EdgeEvidence("schema", _SCHEMA_CONFIDENCE, {"output": output.name, "input": required.name}),
        )
        for source in nodes.values()
        for target in nodes.values()
        if source.name != target.name and focus & {source.name, target.name}
        for required in target.inputs
        if required.required
        for output in source.outputs
        if _same_name(output.name, required.name) and _compatible(output.type, required.type)
    ]


def _same_name(left: str, right: str) -> bool:
    return _NAME_NOISE.sub("", left.lower()) == _NAME_NOISE.sub("", right.lower())


def _compatible(left: str, right: str) -> bool:
    return not left or not right or bool(set(left.split("|")) & set(right.split("|")))


async def _model_proposals(
    nodes: Mapping[str, ToolNode], fresh: Sequence[str], model: Model, meter: UsageMeter
) -> list[_Proposal]:
    async def propose(name: str) -> list[_Proposal]:
        answer = await ask(
            model, DEPENDENCY_SYSTEM_PROMPT, dependency_prompt(nodes[name], nodes.values()), ToolDependencies, meter
        )
        return [
            proposal
            for dependency in answer.dependencies
            if (proposal := _valid_proposal(dependency, name, nodes)) is not None
        ]

    batches = await _bounded([partial(propose, name) for name in fresh], _MODEL_CONCURRENCY)
    return [proposal for batch in batches for proposal in batch]


def _valid_proposal(dependency: ToolDependency, focus: str, nodes: Mapping[str, ToolNode]) -> _Proposal | None:
    source, target, required = dependency.source.strip(), dependency.target.strip(), dependency.input.strip()
    if source == target or focus not in (source, target) or source not in nodes or target not in nodes:
        return None
    if required not in nodes[target].required_inputs:
        return None
    return _Proposal(
        source=source,
        target=target,
        input=required,
        evidence=EdgeEvidence("model", _clamp(dependency.confidence), {"output": dependency.output, "input": required}),
        probe_query=dependency.probe_query.strip(),
    )


async def _probed(
    proposals: list[_Proposal], tools: Mapping[str, AgentTool], model: Model, meter: UsageMeter
) -> list[_Proposal]:
    queries: dict[tuple[str, str], str] = {}
    for proposal in proposals:
        if proposal.evidence.kind == "model" and proposal.probe_query:
            queries.setdefault((proposal.source, proposal.target), proposal.probe_query)
    pairs = list(queries)
    verdicts = await _bounded(
        [
            partial(_probe, tools[source], tools[target], queries[(source, target)], model, meter)
            for source, target in pairs
        ],
        _PROBE_CONCURRENCY,
    )
    outcomes = dict(zip(pairs, verdicts, strict=True))
    return [_with_probe(proposal, queries, outcomes) for proposal in proposals]


def _with_probe(
    proposal: _Proposal,
    queries: Mapping[tuple[str, str], str],
    outcomes: Mapping[tuple[str, str], ProbeVerdict],
) -> _Proposal:
    pair = (proposal.source, proposal.target)
    if proposal.evidence.kind != "model" or pair not in outcomes:
        return proposal
    verdict = outcomes[pair]
    likelihood = _clamp(0.5 + 0.5 * (_clamp(verdict.with_source) - _clamp(verdict.without_source)))
    prior = proposal.evidence.confidence
    detail = {
        **proposal.evidence.detail,
        "query": queries[pair],
        "with_source": _clamp(verdict.with_source),
        "without_source": _clamp(verdict.without_source),
        "prior": prior,
    }
    return replace(proposal, evidence=EdgeEvidence("probe", _posterior(prior, likelihood), detail))


async def _probe(source: AgentTool, target: AgentTool, query: str, model: Model, meter: UsageMeter) -> ProbeVerdict:
    with_source = await _attempt([source, target], query, model, meter)
    without_source = await _attempt([target], query, model, meter)
    return await ask(
        model,
        PROBE_JUDGE_SYSTEM_PROMPT,
        probe_judge_prompt(query, with_source, without_source),
        ProbeVerdict,
        meter,
    )


async def _attempt(tools: list[AgentTool], query: str, model: Model, meter: UsageMeter) -> str:
    agent = Agent(model=model, tools=list(tools), system_prompt=PROBE_SYSTEM_PROMPT, callback_handler=None)
    result = await agent.invoke_async(query, limits={"turns": _PROBE_TURNS})
    meter.add(result.metrics.accumulated_usage)
    return DecisionState.from_messages(agent.messages).transcript


def _merge(proposals: Iterable[_Proposal], min_edge_weight: float) -> list[ToolEdge]:
    grouped: dict[tuple[str, str], list[_Proposal]] = {}
    for proposal in proposals:
        grouped.setdefault((proposal.source, proposal.target), []).append(proposal)
    edges = [
        ToolEdge(
            source=source,
            target=target,
            supplies=tuple(sorted({proposal.input for proposal in group})),
            weight=sum(proposal.evidence.confidence for proposal in group) / len(group),
            evidence=tuple(proposal.evidence for proposal in group),
        )
        for (source, target), group in grouped.items()
    ]
    return [edge for edge in edges if edge.weight > min_edge_weight]


def _posterior(prior: float, likelihood: float) -> float:
    support = prior * likelihood
    total = support + (1 - prior) * (1 - likelihood)
    return prior if total <= 0 else support / total


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


async def _bounded(jobs: Sequence[Callable[[], Awaitable[ResultT]]], limit: int) -> list[ResultT]:
    semaphore = asyncio.Semaphore(limit)

    async def run(job: Callable[[], Awaitable[ResultT]]) -> ResultT:
        async with semaphore:
            return await job()

    return list(await asyncio.gather(*(run(job) for job in jobs)))
