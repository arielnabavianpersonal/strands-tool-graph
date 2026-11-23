import logging

import pytest
from strands import Agent

from strands_tool_graph import ToolGraph, ToolGraphAdvisor, _values
from strands_tool_graph._prompts import CandidateScore, GoalSelection, GraphAdvice, ToolDependencies, ToolDependency
from tests.support import ScriptedModel, call_message, make_tool, shop_graph, shop_tools, text_message

ASK = "Please refund my order. My email is ana@shop.test. The reason is damage."
ADVICE_TAG = "<tool_graph_advice>"


def finds_user_then_finishes(messages, tools):
    has_result = any("toolResult" in block for message in messages for block in message["content"])
    return text_message("done") if has_result else call_message(("find_user", {"email": "ana@shop.test"}))


def answers(goals=("refund_order",), score=8):
    return {
        "GoalSelection": lambda prompt: GoalSelection(goals=list(goals)),
        "CandidateScore": lambda prompt: CandidateScore(score=score, reason="It finds the user id."),
        "GraphAdvice": lambda prompt: GraphAdvice(feedback="Find the user first.", recommended_action="find_user"),
    }


SHOP_DEPENDENCIES = {
    "find_user": ("find_user", "list_orders", "user_id"),
    "list_orders": ("list_orders", "refund_order", "order_id"),
}


def shop_dependencies(prompt):
    focus = prompt.split("Focus tool:\n- ", 1)[1].split(":", 1)[0]
    if focus not in SHOP_DEPENDENCIES:
        return ToolDependencies(dependencies=[])
    source, target, field = SHOP_DEPENDENCIES[focus]
    found = ToolDependency(source=source, target=target, input=field, output=field, confidence=0.9, probe_query="q")
    return ToolDependencies(dependencies=[found])


def focuses(model):
    return [
        call["prompt"].split("Focus tool:\n- ", 1)[1].split(":", 1)[0] for call in model.calls_for("ToolDependencies")
    ]


def advised(call):
    last = call["messages"][-1]
    return [block["text"] for block in last["content"] if ADVICE_TAG in block.get("text", "")]


async def test_advice_reaches_each_model_call_but_not_the_history():
    model = ScriptedModel(plan=finds_user_then_finishes, answers=answers())
    advisor = ToolGraphAdvisor(shop_graph())
    agent = Agent(model=model, tools=shop_tools(), plugins=[advisor], callback_handler=None)

    await agent.invoke_async(ASK)

    real_calls = [call for call in model.stream_calls if advised(call)]
    assert len(real_calls) == 2
    assert len(model.stream_calls) == 2 + 2 * 4
    [first_advice] = advised(real_calls[0])
    assert '1. find_user(email="ana@shop.test") for goal refund_order: score 0.80' in first_advice
    assert "Recommended next action: find_user" in first_advice
    assert ADVICE_TAG not in str(agent.messages)
    assert advisor.usage["totalTokens"] > 0
    assert model.calls_for("ToolDependencies") == []


async def test_goals_are_inferred_once_per_request():
    model = ScriptedModel(plan=finds_user_then_finishes, answers=answers(goals=("refund_order", "drop_table")))
    agent = Agent(model=model, tools=shop_tools(), plugins=[ToolGraphAdvisor(shop_graph())], callback_handler=None)

    await agent.invoke_async(ASK)

    assert len(model.calls_for("GoalSelection")) == 1
    assert "drop_table" not in str(model.calls_for("GraphAdvice"))


async def test_inadmissible_candidates_rank_last_and_skip_the_scorer():
    responses = iter(
        [
            call_message(("list_orders", {"user_id": "u-999"})),
            call_message(("find_user", {"email": "ana@shop.test"})),
            text_message("done"),
        ]
    )
    model = ScriptedModel(plan=lambda messages, tools: next(responses), answers=answers())
    agent = Agent(
        model=model,
        tools=shop_tools(),
        plugins=[ToolGraphAdvisor(shop_graph(), candidates=2)],
        callback_handler=None,
    )

    await agent.invoke_async(ASK)

    [advice] = advised(model.stream_calls[-1])
    rows = [line for line in advice.splitlines() if line[:2] in ("1.", "2.")]
    assert rows[0].startswith('1. find_user(email="ana@shop.test")')
    assert rows[1].startswith('2. list_orders(user_id="u-999")') and "not admissible" in rows[1]
    assert len(model.calls_for("CandidateScore")) == 1


@pytest.mark.parametrize("scorer", ["dp", "pagerank"])
async def test_graph_scorers_make_no_scoring_calls(scorer):
    model = ScriptedModel(plan=finds_user_then_finishes, answers=answers())
    agent = Agent(
        model=model,
        tools=shop_tools(),
        plugins=[ToolGraphAdvisor(shop_graph(), scorer=scorer)],
        callback_handler=None,
    )

    await agent.invoke_async(ASK)

    assert model.calls_for("CandidateScore") == []
    assert len([call for call in model.stream_calls if advised(call)]) == 2


async def test_scorer_samples_are_averaged():
    scores = iter([10, 4, 6, 6])
    model = ScriptedModel(
        plan=finds_user_then_finishes,
        answers={**answers(), "CandidateScore": lambda prompt: CandidateScore(score=next(scores), reason="r")},
    )
    agent = Agent(
        model=model,
        tools=shop_tools(),
        plugins=[ToolGraphAdvisor(shop_graph(), candidates=1, scorer_samples=2)],
        callback_handler=None,
    )

    await agent.invoke_async(ASK)

    [first_advice] = advised(next(call for call in model.stream_calls if advised(call)))
    assert "score 0.70" in first_advice


async def test_a_failed_side_call_leaves_the_model_call_unchanged(caplog):
    def fail(prompt):
        raise RuntimeError("goal model is down")

    model = ScriptedModel(plan=finds_user_then_finishes, answers={**answers(), "GoalSelection": fail})
    agent = Agent(model=model, tools=shop_tools(), plugins=[ToolGraphAdvisor(shop_graph())], callback_handler=None)

    with caplog.at_level(logging.WARNING, logger="strands_tool_graph.advisor"):
        result = await agent.invoke_async(ASK)

    assert str(result).strip() == "done"
    assert not any(advised(call) for call in model.stream_calls)
    assert "goal model is down" in caplog.text


async def test_gives_no_advice_to_an_agent_without_tools():
    model = ScriptedModel(answers=answers())
    agent = Agent(model=model, plugins=[ToolGraphAdvisor(shop_graph())], callback_handler=None)

    await agent.invoke_async(ASK)

    assert len(model.stream_calls) == 1
    assert model.structured_calls == []


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"scorer": "random"}, "scorer"),
        ({"candidates": 0}, "candidates"),
        ({"top_k": 0}, "top_k"),
        ({"hops": -1}, "hops"),
        ({"min_edge_weight": 2.0}, "min_edge_weight"),
        ({"scorer_samples": 0}, "scorer_samples"),
    ],
)
def test_rejects_invalid_settings(settings, message):
    with pytest.raises(ValueError, match=message):
        ToolGraphAdvisor(shop_graph(), **settings)


async def test_an_unsolvable_graph_value_falls_back_to_neutral(monkeypatch):
    monkeypatch.setattr(_values, "_MAX_STATES", 1)
    model = ScriptedModel(plan=finds_user_then_finishes, answers=answers())
    agent = Agent(
        model=model,
        tools=shop_tools(),
        plugins=[ToolGraphAdvisor(shop_graph(), scorer="dp")],
        callback_handler=None,
    )

    await agent.invoke_async(ASK)

    [first_advice] = advised(next(call for call in model.stream_calls if advised(call)))
    assert "score 0.50" in first_advice


async def test_a_built_and_saved_graph_drives_the_advice(tmp_path):
    def dependencies(prompt):
        if "Focus tool:\n- find_user:" not in prompt:
            return ToolDependencies(dependencies=[])
        found = ToolDependency(
            source="find_user", target="list_orders", input="user_id", output="user_id", confidence=0.9, probe_query="q"
        )
        return ToolDependencies(dependencies=[found])

    builder = ScriptedModel(answers={"ToolDependencies": dependencies})
    path = tmp_path / "tool_graph.json"
    (await ToolGraph.build(shop_tools(), model=builder)).save(path)
    model = ScriptedModel(plan=finds_user_then_finishes, answers=answers(goals=("list_orders",)))
    agent = Agent(
        model=model,
        tools=shop_tools(),
        plugins=[ToolGraphAdvisor(ToolGraph.load(path))],
        callback_handler=None,
    )

    await agent.invoke_async(ASK)

    [first_advice] = advised(next(call for call in model.stream_calls if advised(call)))
    assert "find_user -> list_orders (weight 0.90): fills user_id" in str(model.calls_for("GraphAdvice")[0]["prompt"])
    assert 'find_user(email="ana@shop.test") for goal list_orders' in first_advice


async def test_builds_the_graph_from_the_agent_tools_on_the_first_model_call():
    model = ScriptedModel(plan=finds_user_then_finishes, answers={**answers(), "ToolDependencies": shop_dependencies})
    advisor = ToolGraphAdvisor()
    agent = Agent(model=model, tools=shop_tools(), plugins=[advisor], callback_handler=None)

    await agent.invoke_async(ASK)

    assert sorted(focuses(model)) == sorted(tool.tool_name for tool in shop_tools())
    assert [(edge.source, edge.target) for edge in advisor.graph.edges] == [
        ("find_user", "list_orders"),
        ("list_orders", "refund_order"),
    ]
    assert len([call for call in model.stream_calls if advised(call)]) == 2


async def test_updates_the_graph_only_for_tools_added_later():
    model = ScriptedModel(plan=finds_user_then_finishes, answers={**answers(), "ToolDependencies": shop_dependencies})
    advisor = ToolGraphAdvisor()
    agent = Agent(model=model, tools=shop_tools(), plugins=[advisor], callback_handler=None)
    await agent.invoke_async(ASK)
    built = len(focuses(model))

    agent.tool_registry.register_tool(make_tool("cancel_order", "Cancel an order.", {"order_id": "string"}))
    await agent.invoke_async(ASK)

    assert focuses(model)[built:] == ["cancel_order"]
    assert "cancel_order" in advisor.graph.nodes


async def test_the_graph_file_keeps_the_graph_between_advisors(tmp_path):
    path = tmp_path / "graphs" / "tool_graph.json"
    first = ScriptedModel(plan=finds_user_then_finishes, answers={**answers(), "ToolDependencies": shop_dependencies})
    agent = Agent(model=first, tools=shop_tools(), plugins=[ToolGraphAdvisor(graph_file=path)], callback_handler=None)
    await agent.invoke_async(ASK)
    second = ScriptedModel(plan=finds_user_then_finishes, answers=answers())
    advisor = ToolGraphAdvisor(graph_file=path)
    agent = Agent(model=second, tools=shop_tools(), plugins=[advisor], callback_handler=None)

    await agent.invoke_async(ASK)

    assert second.calls_for("ToolDependencies") == []
    assert advisor.graph == ToolGraph.load(path)
    assert len([call for call in second.stream_calls if advised(call)]) == 2


async def test_an_unreadable_graph_file_is_built_again(tmp_path, caplog):
    path = tmp_path / "tool_graph.json"
    path.write_text("not json")
    model = ScriptedModel(plan=finds_user_then_finishes, answers={**answers(), "ToolDependencies": shop_dependencies})
    agent = Agent(model=model, tools=shop_tools(), plugins=[ToolGraphAdvisor(graph_file=path)], callback_handler=None)

    with caplog.at_level(logging.WARNING, logger="strands_tool_graph.advisor"):
        await agent.invoke_async(ASK)

    assert "cannot read the tool graph file" in caplog.text
    assert len(ToolGraph.load(path).edges) == 2
