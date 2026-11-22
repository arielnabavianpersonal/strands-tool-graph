from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from strands.models import Model
    from strands.types.tools import AgentTool

EvidenceKind = Literal["schema", "model", "probe"]

_FORMAT = "strands-tool-graph/1"


@dataclass(frozen=True)
class ToolField:
    name: str
    type: str
    description: str
    required: bool


@dataclass(frozen=True)
class ToolNode:
    name: str
    description: str
    inputs: tuple[ToolField, ...]
    outputs: tuple[ToolField, ...]
    fingerprint: str

    @property
    def required_inputs(self) -> tuple[str, ...]:
        return tuple(tool_field.name for tool_field in self.inputs if tool_field.required)


@dataclass(frozen=True)
class EdgeEvidence:
    kind: EvidenceKind
    confidence: float
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolEdge:
    source: str
    target: str
    supplies: tuple[str, ...]
    weight: float
    evidence: tuple[EdgeEvidence, ...]


@dataclass(frozen=True)
class ToolGraph:
    nodes: Mapping[str, ToolNode]
    edges: tuple[ToolEdge, ...]

    def __post_init__(self) -> None:
        unknown = sorted({name for edge in self.edges for name in (edge.source, edge.target) if name not in self.nodes})
        if unknown:
            raise ValueError(f"edges reference tools that are not in the graph: {', '.join(unknown)}")

    @classmethod
    async def build(
        cls,
        tools: Sequence[AgentTool],
        *,
        model: Model | None = None,
        probe: bool = False,
        min_edge_weight: float = 0.3,
        base: ToolGraph | None = None,
    ) -> ToolGraph:
        from ._builder import build_graph

        return await build_graph(tools, model=model, probe=probe, min_edge_weight=min_edge_weight, base=base)

    @cached_property
    def version(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def incoming(self, name: str) -> tuple[ToolEdge, ...]:
        return self._incoming.get(name, ())

    def outgoing(self, name: str) -> tuple[ToolEdge, ...]:
        return self._outgoing.get(name, ())

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": _FORMAT,
            "nodes": [_node_to_dict(node) for node in self.nodes.values()],
            "edges": [_edge_to_dict(edge) for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ToolGraph:
        if data.get("format") != _FORMAT:
            raise ValueError(f"unsupported tool graph format: {data.get('format')!r}, expected {_FORMAT!r}")
        nodes = [_node_from_dict(node) for node in data["nodes"]]
        return cls(
            nodes={node.name: node for node in nodes},
            edges=tuple(_edge_from_dict(edge) for edge in data["edges"]),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> ToolGraph:
        return cls.from_dict(json.loads(Path(path).read_text()))

    @cached_property
    def _incoming(self) -> dict[str, tuple[ToolEdge, ...]]:
        return _group_edges(self.edges, lambda edge: edge.target)

    @cached_property
    def _outgoing(self) -> dict[str, tuple[ToolEdge, ...]]:
        return _group_edges(self.edges, lambda edge: edge.source)


def _group_edges(edges: Sequence[ToolEdge], key: Callable[[ToolEdge], str]) -> dict[str, tuple[ToolEdge, ...]]:
    grouped: dict[str, list[ToolEdge]] = {}
    for edge in edges:
        grouped.setdefault(key(edge), []).append(edge)
    return {name: tuple(group) for name, group in grouped.items()}


def _field_to_dict(tool_field: ToolField) -> dict[str, Any]:
    return {
        "name": tool_field.name,
        "type": tool_field.type,
        "description": tool_field.description,
        "required": tool_field.required,
    }


def _field_from_dict(data: Mapping[str, Any]) -> ToolField:
    return ToolField(
        name=data["name"],
        type=data.get("type", ""),
        description=data.get("description", ""),
        required=bool(data.get("required", False)),
    )


def _node_to_dict(node: ToolNode) -> dict[str, Any]:
    return {
        "name": node.name,
        "description": node.description,
        "inputs": [_field_to_dict(tool_field) for tool_field in node.inputs],
        "outputs": [_field_to_dict(tool_field) for tool_field in node.outputs],
        "fingerprint": node.fingerprint,
    }


def _node_from_dict(data: Mapping[str, Any]) -> ToolNode:
    return ToolNode(
        name=data["name"],
        description=data.get("description", ""),
        inputs=tuple(_field_from_dict(tool_field) for tool_field in data.get("inputs", ())),
        outputs=tuple(_field_from_dict(tool_field) for tool_field in data.get("outputs", ())),
        fingerprint=data.get("fingerprint", ""),
    )


def _edge_to_dict(edge: ToolEdge) -> dict[str, Any]:
    return {
        "source": edge.source,
        "target": edge.target,
        "supplies": list(edge.supplies),
        "weight": edge.weight,
        "evidence": [
            {"kind": evidence.kind, "confidence": evidence.confidence, "detail": dict(evidence.detail)}
            for evidence in edge.evidence
        ],
    }


def _edge_from_dict(data: Mapping[str, Any]) -> ToolEdge:
    return ToolEdge(
        source=data["source"],
        target=data["target"],
        supplies=tuple(data.get("supplies", ())),
        weight=float(data["weight"]),
        evidence=tuple(
            EdgeEvidence(kind=evidence["kind"], confidence=float(evidence["confidence"]), detail=evidence["detail"])
            for evidence in data.get("evidence", ())
        ),
    )
