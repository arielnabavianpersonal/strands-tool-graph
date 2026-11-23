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
from typing import TYPE_CHECKING, Literal, cast

from strands.event_loop.streaming import stream_messages
from strands.middleware import InvokeModelContext, InvokeModelStage
from strands.plugins import Plugin
from strands.types.content import ContentBlock, Message, split_system_prompt
from strands.types.event_loop import Usage

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


class ToolGraphAdvisor(Plugin):
    name = "tool-graph-advisor"

    def __init__(
        self,
        graph: ToolGraph,
        *,
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
        self._graph = graph
        self._scorer = scorer
        self._candidate_count = candidates
        self._top_k = top_k
        self._hops = hops
        self._min_edge_weight = min_edge_weight
        self._scorer_samples = scorer_samples
        self._model = model
        self._meter = UsageMeter()
        self._goal_lock = threading.Lock()
        self._goal_cache: OrderedDict[str, tuple[str, ...]] = OrderedDict()

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
        side_model = self._model or getattr(context.agent, "aux_model", None) or context.model
        goals, candidates = await asyncio.gather(self._goals(state, side_model), self._candidates(context))
        if not candidates:
            return None
        values = DependencyValues(self._graph, state)
        alternatives = values.feasible(goals, _MAX_ALTERNATIVES)
        priors = {candidate.key: {goal: _prior(values, candidate, goal) for goal in goals} for candidate in candidates}
        admissible = {candidate.key: values.admissible(candidate) for candidate in candidates}
        policy = split_system_prompt(context.system_prompt)[0] or ""
        scores = await self._scores(state, candidates, goals, priors, admissible, alternatives, side_model, policy)
        rows = _ranked(candidates, goals, priors, scores, self._scorer)[: self._top_k]
        lines = [_describe_row(index, row) for index, row in enumerate(rows, 1)]
        tools = {tool for row in rows for tool in row.candidate.tools}
        graph_evidence = evidence(self._graph, state, tools, goals, self._hops, self._min_edge_weight)
        advice = await ask(
            side_model,
            VERBALIZER_SYSTEM_PROMPT,
            verbalizer_prompt(state.request, lines, alternatives, graph_evidence),
            GraphAdvice,
            self._meter,
        )
        return _render(lines, advice)

    async def _goals(self, state: DecisionState, model: Model) -> tuple[str, ...]:
        if not state.request.strip():
            return ()
        key = hashlib.sha256(state.request.encode()).hexdigest()
        with self._goal_lock:
            cached = self._goal_cache.get(key)
            if cached is not None:
                self._goal_cache.move_to_end(key)
        if cached is not None:
            return cached
        selection = await ask(
            model, GOALS_SYSTEM_PROMPT, goals_prompt(state.request, self._graph), GoalSelection, self._meter
        )
        names = (name.strip() for name in selection.goals)
        goals = tuple(dict.fromkeys(name for name in names if name in self._graph.nodes))[:_MAX_GOALS]
        with self._goal_lock:
            self._goal_cache[key] = goals
            while len(self._goal_cache) > _GOAL_CACHE_SIZE:
                self._goal_cache.popitem(last=False)
        return goals

    async def _candidates(self, context: InvokeModelContext) -> list[Candidate]:
        if getattr(context.model, "stateful", False):
            return []
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

    async def _scores(
        self,
        state: DecisionState,
        candidates: Sequence[Candidate],
        goals: Sequence[str],
        priors: Mapping[str, Mapping[str, float | None]],
        admissible: Mapping[str, bool],
        alternatives: Sequence[str],
        model: Model,
        policy: str,
    ) -> dict[str, float | None]:
        if self._scorer == "pagerank":
            ranks = pagerank_scores(self._graph, state.request)
            return {
                candidate.key: _pagerank_score(candidate, ranks) if admissible[candidate.key] else None
                for candidate in candidates
            }
        if self._scorer == "dp":
            return {candidate.key: NEUTRAL if admissible[candidate.key] else None for candidate in candidates}
        scored = [candidate for candidate in candidates if admissible[candidate.key]]
        results = await asyncio.gather(
            *(
                self._llm_score(state, candidate, priors[candidate.key], goals, alternatives, model, policy)
                for candidate in scored
            )
        )
        scores: dict[str, float | None] = {candidate.key: None for candidate in candidates}
        scores.update({candidate.key: score for candidate, score in zip(scored, results, strict=True)})
        return scores

    async def _llm_score(
        self,
        state: DecisionState,
        candidate: Candidate,
        priors: Mapping[str, float | None],
        goals: Sequence[str],
        alternatives: Sequence[str],
        model: Model,
        policy: str,
    ) -> float:
        graph_evidence = evidence(self._graph, state, candidate.tools, goals, self._hops, self._min_edge_weight)
        prompt = scorer_prompt(state, candidate, priors, alternatives, graph_evidence, policy)
        answers = await asyncio.gather(
            *(
                ask(model, SCORER_SYSTEM_PROMPT, prompt, CandidateScore, self._meter)
                for _ in range(self._scorer_samples)
            )
        )
        return sum(max(0, min(_SCORE_SCALE, answer.score)) for answer in answers) / (len(answers) * _SCORE_SCALE)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _prior(values: DependencyValues, candidate: Candidate, goal: str) -> float | None:
    try:
        return values.candidate_value(candidate, goal)
    except ValueIterationError as error:
        logger.debug("goal=<%s>, error=<%s> | graph value is unknown | using a neutral value", goal, error)
        return NEUTRAL


def _needs_advice(context: InvokeModelContext) -> bool:
    forced = isinstance(context.tool_choice, Mapping) and "tool" in context.tool_choice
    return bool(context.tool_specs) and bool(context.messages) and context.messages[-1]["role"] == "user" and not forced


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
