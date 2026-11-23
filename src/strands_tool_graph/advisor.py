from __future__ import annotations

import asyncio
import hashlib
import html
import logging
import threading
import warnings
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from strands.event_loop.streaming import stream_messages
from strands.middleware import InvokeModelContext, InvokeModelStage
from strands.plugins import Plugin
from strands.types.content import ContentBlock, Message, split_system_prompt
from strands.types.event_loop import Usage

from ._builder import catalog_key, graph_key
from ._llm import StreamUsage, UsageMeter, ask
from ._prompts import (
    GOALS_SYSTEM_PROMPT,
    SCORER_SYSTEM_PROMPT,
    VERBALIZER_SYSTEM_PROMPT,
    CandidateScore,
    GoalSelection,
    GraphAdvice,
    evidence,
    goals_prompt,
    scorer_prompt,
    verbalizer_prompt,
)
from ._state import Candidate, DecisionState
from ._values import NEUTRAL, DependencyValues, ValueIterationError, pagerank_scores
from .graph import ToolGraph

if TYPE_CHECKING:
    from strands.agent.agent import Agent
    from strands.models import Model
    from strands.types.tools import AgentTool

Scorer = Literal["llm", "dp", "pagerank"]

_SCORERS: tuple[Scorer, ...] = ("llm", "dp", "pagerank")
_MAX_GOALS = 3
_MAX_ALTERNATIVES = 16
_GOAL_CACHE_SIZE = 64
_SCORE_SCALE = 10
_HEADER = "Advice from a dependency graph of your tools. You still choose the next action."

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Row:
    candidate: Candidate
    goal: str | None
    score: float | None
    prior: float | None


@dataclass(frozen=True)
class _Decision:
    graph: ToolGraph
    state: DecisionState
    goals: tuple[str, ...]
    priors: Mapping[str, Mapping[str, float | None]]
    admissible: Mapping[str, bool]
    alternatives: tuple[str, ...]
    model: Model
    policy: str


class ToolGraphAdvisor(Plugin):
    name = "tool-graph-advisor"

    def __init__(
        self,
        graph: ToolGraph | None = None,
        *,
        graph_file: str | Path | None = None,
        probe: bool = False,
        scorer: Scorer = "llm",
        candidates: int = 4,
        top_k: int = 8,
        hops: int = 1,
        min_edge_weight: float = 0.0,
        scorer_samples: int = 1,
        model: Model | None = None,
    ) -> None:
        _require(scorer in _SCORERS, f"scorer must be one of {', '.join(_SCORERS)}, got {scorer!r}")
        _require(candidates >= 1, f"candidates must be at least 1, got {candidates}")
        _require(top_k >= 1, f"top_k must be at least 1, got {top_k}")
        _require(hops >= 0, f"hops must be 0 or more, got {hops}")
        _require(0 <= min_edge_weight <= 1, f"min_edge_weight must be between 0 and 1, got {min_edge_weight}")
        _require(scorer_samples >= 1, f"scorer_samples must be at least 1, got {scorer_samples}")
        super().__init__()
        self._graph_file = Path(graph_file) if graph_file is not None else None
        self._probe = probe
        self._scorer = scorer
        self._candidate_count = candidates
        self._top_k = top_k
        self._hops = hops
        self._min_edge_weight = min_edge_weight
        self._scorer_samples = scorer_samples
        self._model = model
        self._meter = UsageMeter()
        self._lock = threading.Lock()
        self._latest = graph
        self._graphs: dict[str, ToolGraph] = {}
        self._goal_cache: OrderedDict[str, tuple[str, ...]] = OrderedDict()

    @property
    def graph(self) -> ToolGraph | None:
        with self._lock:
            return self._latest

    @property
    def usage(self) -> Usage:
        return self._meter.usage

    def init_agent(self, agent: Agent) -> None:
        if getattr(agent.model, "stateful", False):
            warnings.warn(
                "ToolGraphAdvisor gives no advice to a model that keeps the conversation on the server",
                stacklevel=2,
            )
        agent.add_middleware(InvokeModelStage.Input, self._advise)

    async def _advise(self, context: InvokeModelContext) -> InvokeModelContext:
        if not _needs_advice(context):
            return context
        try:
            text = await self._advice(context)
        except Exception as error:
            logger.warning("error=<%s> | tool graph advice failed | the model call proceeds without advice", error)
            return context
        return _with_advice(context, text) if text else context

    async def _advice(self, context: InvokeModelContext) -> str | None:
        state = DecisionState.from_messages(context.messages)
        model = self._model or getattr(context.agent, "aux_model", None) or context.model
        (graph, goals), candidates = await asyncio.gather(
            self._graph_and_goals(context, state, model), self._candidates(context)
        )
        if not candidates:
            return None
        values = DependencyValues(graph, state)
        decision = _Decision(
            graph=graph,
            state=state,
            goals=goals,
            priors={
                candidate.key: {goal: _prior(values, candidate, goal) for goal in goals} for candidate in candidates
            },
            admissible={candidate.key: values.admissible(candidate) for candidate in candidates},
            alternatives=values.feasible(goals, _MAX_ALTERNATIVES),
            model=model,
            policy=split_system_prompt(context.system_prompt)[0] or "",
        )
        scores = await self._scores(decision, candidates)
        rows = _ranked(candidates, goals, decision.priors, scores, self._scorer)[: self._top_k]
        lines = [_describe_row(index, row) for index, row in enumerate(rows, 1)]
        tools = {tool for row in rows for tool in row.candidate.tools}
        graph_evidence = evidence(graph, state, tools, goals, self._hops, self._min_edge_weight)
        advice = await ask(
            model,
            VERBALIZER_SYSTEM_PROMPT,
            verbalizer_prompt(state.request, lines, decision.alternatives, graph_evidence),
            GraphAdvice,
            self._meter,
        )
        return _render(lines, advice)

    async def _graph_and_goals(
        self, context: InvokeModelContext, state: DecisionState, model: Model
    ) -> tuple[ToolGraph, tuple[str, ...]]:
        graph = await self._graph_for(context, model)
        return graph, await self._goals(graph, state, model)

    async def _graph_for(self, context: InvokeModelContext, model: Model) -> ToolGraph:
        tools = _registered_tools(context)
        key = catalog_key(tools)
        with self._lock:
            known = self._graphs.get(key)
            base = self._latest
        if known is not None:
            return known
        base = base if base is not None else self._load_file()
        if base is not None and graph_key(base) == key:
            graph = base
        else:
            logger.info("tools=<%d> | building the tool graph for the agent tools", len(tools))
            graph = await ToolGraph.build(tools, model=model, probe=self._probe, base=base)
            self._save_file(graph)
        with self._lock:
            self._graphs[key] = graph
            self._latest = graph
        return graph

    def _load_file(self) -> ToolGraph | None:
        if self._graph_file is None or not self._graph_file.exists():
            return None
        try:
            return ToolGraph.load(self._graph_file)
        except (OSError, ValueError, KeyError, TypeError) as error:
            logger.warning(
                "path=<%s>, error=<%s> | cannot read the tool graph file | building a new graph",
                self._graph_file,
                error,
            )
            return None

    def _save_file(self, graph: ToolGraph) -> None:
        if self._graph_file is None:
            return
        try:
            graph.save(self._graph_file)
        except OSError as error:
            logger.warning("path=<%s>, error=<%s> | cannot write the tool graph file", self._graph_file, error)

    async def _goals(self, graph: ToolGraph, state: DecisionState, model: Model) -> tuple[str, ...]:
        if not state.request.strip():
            return ()
        key = hashlib.sha256(f"{graph.version}\n{state.request}".encode()).hexdigest()
        with self._lock:
            cached = self._goal_cache.get(key)
            if cached is not None:
                self._goal_cache.move_to_end(key)
        if cached is not None:
            return cached
        selection = await ask(
            model, GOALS_SYSTEM_PROMPT, goals_prompt(state.request, graph), GoalSelection, self._meter
        )
        names = (name.strip() for name in selection.goals)
        goals = tuple(dict.fromkeys(name for name in names if name in graph.nodes))[:_MAX_GOALS]
        with self._lock:
            self._goal_cache[key] = goals
            while len(self._goal_cache) > _GOAL_CACHE_SIZE:
                self._goal_cache.popitem(last=False)
        return goals

    async def _candidates(self, context: InvokeModelContext) -> list[Candidate]:
        sampled = await asyncio.gather(*(self._sample(context) for _ in range(self._candidate_count)))
        unique: dict[str, Candidate] = {}
        for candidate in sampled:
            if candidate is not None:
                unique.setdefault(candidate.key, candidate)
        return list(unique.values())

    async def _sample(self, context: InvokeModelContext) -> Candidate | None:
        system_prompt, system_prompt_content = split_system_prompt(context.system_prompt)
        usage = StreamUsage()
        message: Message | None = None
        async for event in stream_messages(
            context.model,
            system_prompt,
            context.messages,
            context.tool_specs,
            tool_choice=context.tool_choice,
            system_prompt_content=system_prompt_content,
            dynamic_trailing_blocks=context.dynamic_trailing_blocks,
        ):
            usage.observe(event)
            stop = event.get("stop")
            if stop:
                message = stop[1]
        self._meter.add(usage.total)
        if message is None:
            return None
        candidate = Candidate.from_message(message)
        return candidate if candidate.calls or candidate.text else None

    async def _scores(self, decision: _Decision, candidates: Sequence[Candidate]) -> dict[str, float | None]:
        if self._scorer == "pagerank":
            ranks = pagerank_scores(decision.graph, decision.state.request)
            return {
                candidate.key: _pagerank_score(candidate, ranks) if decision.admissible[candidate.key] else None
                for candidate in candidates
            }
        if self._scorer == "dp":
            return {candidate.key: NEUTRAL if decision.admissible[candidate.key] else None for candidate in candidates}
        scored = [candidate for candidate in candidates if decision.admissible[candidate.key]]
        results = await asyncio.gather(*(self._llm_score(decision, candidate) for candidate in scored))
        scores: dict[str, float | None] = {candidate.key: None for candidate in candidates}
        scores.update({candidate.key: score for candidate, score in zip(scored, results, strict=True)})
        return scores

    async def _llm_score(self, decision: _Decision, candidate: Candidate) -> float:
        graph_evidence = evidence(
            decision.graph, decision.state, candidate.tools, decision.goals, self._hops, self._min_edge_weight
        )
        prompt = scorer_prompt(
            decision.state,
            candidate,
            decision.priors[candidate.key],
            decision.alternatives,
            graph_evidence,
            decision.policy,
        )
        answers = await asyncio.gather(
            *(
                ask(decision.model, SCORER_SYSTEM_PROMPT, prompt, CandidateScore, self._meter)
                for _ in range(self._scorer_samples)
            )
        )
        return sum(max(0, min(_SCORE_SCALE, answer.score)) for answer in answers) / (len(answers) * _SCORE_SCALE)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _registered_tools(context: InvokeModelContext) -> list[AgentTool]:
    registry = context.agent.tool_registry
    tools = dict(registry.dynamic_tools)
    tools.update(registry.registry)
    return list(tools.values())


def _prior(values: DependencyValues, candidate: Candidate, goal: str) -> float | None:
    try:
        return values.candidate_value(candidate, goal)
    except ValueIterationError as error:
        logger.debug("goal=<%s>, error=<%s> | graph value is unknown | using a neutral value", goal, error)
        return NEUTRAL


def _needs_advice(context: InvokeModelContext) -> bool:
    forced = isinstance(context.tool_choice, Mapping) and "tool" in context.tool_choice
    stateful = getattr(context.model, "stateful", False)
    has_turn = bool(context.messages) and context.messages[-1]["role"] == "user"
    return bool(context.tool_specs) and has_turn and not forced and not stateful


def _pagerank_score(candidate: Candidate, ranks: Mapping[str, float]) -> float:
    if not candidate.calls:
        return NEUTRAL
    return sum(ranks.get(tool, 0.0) for tool in candidate.tools) / len(candidate.tools)


def _ranked(
    candidates: Sequence[Candidate],
    goals: Sequence[str],
    priors: Mapping[str, Mapping[str, float | None]],
    scores: Mapping[str, float | None],
    scorer: Scorer,
) -> list[_Row]:
    rows = [
        _Row(
            candidate=candidate,
            goal=goal,
            score=priors[candidate.key][goal] if scorer == "dp" and goal is not None else scores[candidate.key],
            prior=priors[candidate.key][goal] if goal is not None else None,
        )
        for candidate in candidates
        for goal in (goals or (None,))
    ]
    return sorted(rows, key=lambda row: (row.score is not None, row.score or 0.0, row.prior or 0.0), reverse=True)


def _describe_row(index: int, row: _Row) -> str:
    action = html.escape(row.candidate.describe(), quote=False)
    goal = f" for goal {row.goal}" if row.goal else ""
    if row.score is None:
        return f"{index}. {action}{goal}: not admissible (a required input is missing or not grounded)"
    prior = "" if row.prior is None else f" (graph value {row.prior:.2f})"
    return f"{index}. {action}{goal}: score {row.score:.2f}{prior}"


def _render(lines: Sequence[str], advice: GraphAdvice) -> str:
    return "\n".join(
        [
            "<tool_graph_advice>",
            _HEADER,
            "Ranked options:",
            *lines,
            f"Feedback: {html.escape(advice.feedback.strip(), quote=False)}",
            f"Recommended next action: {html.escape(advice.recommended_action.strip(), quote=False)}",
            "</tool_graph_advice>",
        ]
    )


def _with_advice(context: InvokeModelContext, text: str) -> InvokeModelContext:
    messages = list(context.messages)
    last = messages[-1]
    separator = "\n\n" if any("text" in block for block in last["content"]) else ""
    block: ContentBlock = {"text": f"{separator}{text}"}
    messages[-1] = cast(Message, {**last, "content": [*last["content"], block]})
    return context.replace(messages=messages, dynamic_trailing_blocks=context.dynamic_trailing_blocks + 1)
