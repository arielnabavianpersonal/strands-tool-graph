from __future__ import annotations

import heapq
import math
import re
from collections import Counter, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property

from ._state import Candidate, DecisionState, ToolCall
from .graph import ToolGraph

DISCOUNT = 0.95
CONFIDENCE_PENALTY = 0.3
NEUTRAL = 0.5
_TOLERANCE = 1e-9
_MAX_ITERATIONS = 10_000
_MAX_STATES = 100_000
_DAMPING = 0.85
_PAGERANK_ITERATIONS = 100
_PAGERANK_TOLERANCE = 1e-10
_TOKEN = re.compile(r"[a-z0-9]+")


class ValueIterationError(RuntimeError):
    pass


@dataclass(frozen=True)
class _Action:
    requires: frozenset[str]
    produces: frozenset[str]
    success: float
    confidence: float


class DependencyValues:
    def __init__(self, graph: ToolGraph, state: DecisionState) -> None:
        self._graph = graph
        self._state = state
        self._ancestor_cache: dict[str, frozenset[str]] = {}
        self._goal_cache: dict[str, _GoalValues] = {}

    @cached_property
    def entities(self) -> frozenset[str]:
        return frozenset(
            entity for name in self._state.succeeded if name in self._actions for entity in self._actions[name].produces
        )

    def feasible(self, goals: Iterable[str], limit: int) -> tuple[str, ...]:
        relevant = {name for goal in goals for name in (goal, *self._ancestors(goal))}
        executable = [
            name for name in self._graph.nodes if name in relevant and self._actions[name].requires <= self.entities
        ]
        return tuple(executable[:limit])

    def admissible(self, candidate: Candidate) -> bool:
        return not candidate.calls or self._composite(candidate.calls) is not None

    def candidate_value(self, candidate: Candidate, goal: str) -> float | None:
        goal_values = self._goal_values(goal)
        if not candidate.calls:
            return _to_unit(DISCOUNT * goal_values.value(self.entities))
        composite = self._composite(candidate.calls)
        if composite is None:
            return None
        if goal in candidate.tools:
            return 1.0
        action, start = composite
        before = goal_values.value(start)
        after = goal_values.value(start | action.produces)
        raw = DISCOUNT * (action.success * after + (1 - action.success) * before)
        return _to_unit(_shrink(raw, action.confidence))

    def _composite(self, calls: tuple[ToolCall, ...]) -> tuple[_Action, frozenset[str]] | None:
        grounded: set[str] = set()
        for call in calls:
            entities = self._grounded(call)
            if entities is None:
                return None
            grounded |= entities
        actions = [self._actions[call.name] for call in calls]
        composite = _Action(
            requires=frozenset(),
            produces=frozenset().union(*(action.produces for action in actions)),
            success=math.prod(action.success for action in actions),
            confidence=min(action.confidence for action in actions),
        )
        return composite, self.entities | grounded

    def _grounded(self, call: ToolCall) -> frozenset[str] | None:
        if call.name not in self._graph.nodes:
            return None
        producible = self._producible(call.name)
        grounded = set()
        for name in self._graph.nodes[call.name].required_inputs:
            entity = f"{call.name}.{name}"
            supplied = call.input.get(name) not in (None, "")
            if entity not in producible:
                if not supplied:
                    return None
            elif entity not in self.entities:
                if not supplied or not self._state.grounds(call.input[name]):
                    return None
                grounded.add(entity)
        return frozenset(grounded)

    def _goal_values(self, goal: str) -> _GoalValues:
        if goal not in self._goal_cache:
            ancestors = self._ancestors(goal)
            relevant = frozenset(entity for name in (goal, *ancestors) for entity in self._actions[name].requires)
            actions = tuple(
                _Action(action.requires, action.produces & relevant, action.success, action.confidence)
                for name, action in self._actions.items()
                if name in ancestors
            )
            self._goal_cache[goal] = _GoalValues(actions, relevant, self._actions[goal].requires)
        return self._goal_cache[goal]

    def _ancestors(self, goal: str) -> frozenset[str]:
        if goal not in self._ancestor_cache:
            seen: set[str] = set()
            queue = deque([goal])
            while queue:
                for edge in self._graph.incoming(queue.popleft()):
                    if edge.source not in seen and edge.source != goal:
                        seen.add(edge.source)
                        queue.append(edge.source)
            self._ancestor_cache[goal] = frozenset(seen)
        return self._ancestor_cache[goal]

    def _producible(self, name: str) -> frozenset[str]:
        return frozenset(f"{name}.{supplied}" for edge in self._graph.incoming(name) for supplied in edge.supplies)

    @cached_property
    def _actions(self) -> dict[str, _Action]:
        return {name: self._action(name) for name in self._graph.nodes}

    def _action(self, name: str) -> _Action:
        weights = [edge.weight for edge in self._graph.incoming(name)]
        required = {f"{name}.{field}" for field in self._graph.nodes[name].required_inputs}
        return _Action(
            requires=frozenset(required) & self._producible(name),
            produces=frozenset(
                f"{edge.target}.{supplied}" for edge in self._graph.outgoing(name) for supplied in edge.supplies
            ),
            success=0.0 if name in self._state.failed_only else (max(weights) if weights else NEUTRAL),
            confidence=sum(weights) / len(weights) if weights else NEUTRAL,
        )


class _GoalValues:
    def __init__(self, actions: tuple[_Action, ...], relevant: frozenset[str], requires: frozenset[str]) -> None:
        self._actions = actions
        self._relevant = relevant
        self._requires = requires
        self._lattices: list[dict[frozenset[str], float]] = []

    def value(self, entities: frozenset[str]) -> float:
        start = entities & self._relevant
        for lattice in self._lattices:
            if start in lattice:
                return lattice[start]
        lattice = _solve(self._actions, self._requires, start)
        self._lattices.append(lattice)
        return lattice[start]


def _solve(
    actions: tuple[_Action, ...], goal_requires: frozenset[str], start: frozenset[str]
) -> dict[frozenset[str], float]:
    values: dict[frozenset[str], float] = {}
    for state in sorted(_reachable(actions, goal_requires, start), key=len, reverse=True):
        values[state] = 1.0 if goal_requires <= state else _fixed_point(actions, state, values)
    return values


def _reachable(
    actions: tuple[_Action, ...], goal_requires: frozenset[str], start: frozenset[str]
) -> set[frozenset[str]]:
    seen = {start}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        if goal_requires <= state:
            continue
        for action in actions:
            after = state | action.produces
            if action.requires <= state and after not in seen:
                seen.add(after)
                queue.append(after)
        if len(seen) > _MAX_STATES:
            raise ValueIterationError(f"more than {_MAX_STATES} entity states are reachable")
    return seen


def _fixed_point(actions: tuple[_Action, ...], state: frozenset[str], values: Mapping[frozenset[str], float]) -> float:
    executable = [action for action in actions if action.requires <= state]
    if not executable:
        return -1.0
    successors = [values.get(state | action.produces) for action in executable]
    value = 0.0
    for _ in range(_MAX_ITERATIONS):
        updated = max(
            _shrink(
                DISCOUNT * (action.success * (value if after is None else after) + (1 - action.success) * value),
                action.confidence,
            )
            for action, after in zip(executable, successors, strict=True)
        )
        if abs(updated - value) < _TOLERANCE:
            return updated
        value = updated
    raise ValueIterationError(f"value iteration did not converge in {_MAX_ITERATIONS} steps")


def _shrink(raw: float, confidence: float) -> float:
    return max(-1.0, min(1.0, raw - CONFIDENCE_PENALTY * (1 - confidence) * abs(raw)))


def _to_unit(value: float) -> float:
    return (value + 1) / 2


def pagerank_scores(graph: ToolGraph, request: str) -> dict[str, float]:
    if not graph.nodes:
        return {}
    personalization = _relevance(graph, request)
    ranks = dict(personalization)
    for _ in range(_PAGERANK_ITERATIONS):
        updated = _pagerank_step(graph, ranks, personalization)
        change = sum(abs(updated[name] - ranks[name]) for name in ranks)
        ranks = updated
        if change < _PAGERANK_TOLERANCE:
            break
    top = max(ranks.values()) or 1.0
    return {name: rank / top for name, rank in ranks.items()}


def _pagerank_step(
    graph: ToolGraph, ranks: Mapping[str, float], personalization: Mapping[str, float]
) -> dict[str, float]:
    updated = {name: (1 - _DAMPING) * personalization[name] for name in graph.nodes}
    dangling = 0.0
    for name, rank in ranks.items():
        edges = [edge for edge in graph.outgoing(name) if edge.weight > 0]
        total = sum(edge.weight for edge in edges)
        if not edges:
            dangling += rank
        for edge in edges:
            updated[edge.target] += _DAMPING * rank * edge.weight / total
    return {name: rank + _DAMPING * dangling * personalization[name] for name, rank in updated.items()}


def _relevance(graph: ToolGraph, request: str) -> dict[str, float]:
    documents = {name: Counter(_tokens(f"{name} {node.description}")) for name, node in graph.nodes.items()}
    frequency = Counter(token for document in documents.values() for token in document)
    weight = {token: math.log((len(documents) + 1) / (count + 1)) + 1 for token, count in frequency.items()}
    query = Counter(_tokens(request))
    scores = {name: _cosine(query, document, weight) for name, document in documents.items()}
    total = sum(scores.values())
    if total <= 0:
        return {name: 1 / len(documents) for name in documents}
    return {name: score / total for name, score in scores.items()}


def _cosine(query: Counter[str], document: Counter[str], weight: Mapping[str, float]) -> float:
    dot = sum(query[token] * document[token] * weight.get(token, 0.0) ** 2 for token in query)
    if dot <= 0:
        return 0.0
    query_norm = math.sqrt(sum((query[token] * weight.get(token, 0.0)) ** 2 for token in query))
    document_norm = math.sqrt(sum((document[token] * weight[token]) ** 2 for token in document))
    return dot / (query_norm * document_norm)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.replace("_", " ").lower())


def neighborhood(graph: ToolGraph, seeds: Iterable[str], hops: int, min_edge_weight: float) -> frozenset[str]:
    frontier = {seed for seed in seeds if seed in graph.nodes}
    seen = set(frontier)
    for _ in range(hops):
        frontier = {
            other
            for name in frontier
            for edge in (*graph.incoming(name), *graph.outgoing(name))
            if edge.weight >= min_edge_weight
            for other in (edge.source, edge.target)
            if other not in seen
        }
        seen |= frontier
    return frozenset(seen)


def best_path(
    graph: ToolGraph, source: str, target: str, min_edge_weight: float
) -> tuple[tuple[str, ...], float] | None:
    if source == target:
        return (source,), 1.0
    distances = {source: 0.0}
    previous: dict[str, str] = {}
    heap = [(0.0, source)]
    while heap:
        distance, name = heapq.heappop(heap)
        if name == target:
            return _unwind(previous, target), math.exp(-distance)
        if distance > distances[name]:
            continue
        for edge in graph.outgoing(name):
            if edge.weight <= 0 or edge.weight < min_edge_weight:
                continue
            candidate = distance - math.log(edge.weight)
            if candidate < distances.get(edge.target, math.inf):
                distances[edge.target] = candidate
                previous[edge.target] = name
                heapq.heappush(heap, (candidate, edge.target))
    return None


def _unwind(previous: Mapping[str, str], target: str) -> tuple[str, ...]:
    path = [target]
    while path[-1] in previous:
        path.append(previous[path[-1]])
    return tuple(reversed(path))
