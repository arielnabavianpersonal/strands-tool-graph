import pytest

from strands_tool_graph import ToolGraph
from strands_tool_graph._prompts import ProbeVerdict, ToolDependencies, ToolDependency
from tests.support import ScriptedModel, make_tool, shop_tools, text_message


def dependency(source, target, field, confidence=0.6, query="Refund my last order, my email is a@b.co"):
    return ToolDependency(
        source=source, target=target, input=field, output=field, confidence=confidence, probe_query=query
    )


def focus_of(prompt):
    return prompt.split("Focus tool:\n- ", 1)[1].split(":", 1)[0]


def answering(by_focus):
    return {"ToolDependencies": lambda prompt: ToolDependencies(dependencies=by_focus.get(focus_of(prompt), []))}


async def test_schema_match_links_a_declared_output_to_a_required_input():
    tools = [
        make_tool("find_user", "Find a customer.", {"email": "string"}, outputs={"userId": "string"}),
        make_tool("list_orders", "List orders.", {"user_id": "string"}),
    ]
    model = ScriptedModel(answers=answering({}))

    graph = await ToolGraph.build(tools, model=model)

    [found] = graph.edges
    assert (found.source, found.target, found.supplies, found.weight) == ("find_user", "list_orders", ("user_id",), 1.0)
    assert found.evidence[0].kind == "schema"


async def test_model_proposals_from_both_ends_are_averaged_into_one_edge():
    model = ScriptedModel(
        answers=answering(
            {
                "find_user": [dependency("find_user", "list_orders", "user_id", 0.9)],
                "list_orders": [dependency("find_user", "list_orders", "user_id", 0.5)],
            }
        )
    )

    graph = await ToolGraph.build(shop_tools(), model=model)

    [found] = graph.edges
    assert found.weight == pytest.approx(0.7)
    assert [evidence.kind for evidence in found.evidence] == ["model", "model"]
    assert len(model.calls_for("ToolDependencies")) == len(shop_tools())


async def test_drops_proposals_that_do_not_fit_the_catalog():
    model = ScriptedModel(
        answers=answering(
            {
                "find_user": [
                    dependency("find_user", "delete_account", "user_id"),
                    dependency("find_user", "list_orders", "limit"),
                    dependency("find_user", "find_user", "email"),
                    dependency("list_orders", "refund_order", "order_id"),
                ]
            }
        )
    )

    graph = await ToolGraph.build(shop_tools(), model=model)

    assert graph.edges == ()


async def test_prunes_edges_at_or_below_the_minimum_weight():
    model = ScriptedModel(
        answers=answering(
            {
                "find_user": [dependency("find_user", "list_orders", "user_id", 0.3)],
                "list_orders": [dependency("list_orders", "refund_order", "order_id", 0.31)],
            }
        )
    )

    graph = await ToolGraph.build(shop_tools(), model=model, min_edge_weight=0.3)

    assert [(found.source, found.target) for found in graph.edges] == [("list_orders", "refund_order")]


async def test_rebuild_analyzes_only_new_and_changed_tools():
    first = ScriptedModel(
        answers=answering(
            {
                "find_user": [dependency("find_user", "list_orders", "user_id", 0.9)],
                "list_orders": [dependency("list_orders", "refund_order", "order_id", 0.8)],
            }
        )
    )
    base = await ToolGraph.build(shop_tools(), model=first)
    catalog = [tool for tool in shop_tools() if tool.tool_name != "refund_order"]
    catalog.append(make_tool("cancel_order", "Cancel an order.", {"order_id": "string"}))
    second = ScriptedModel(answers=answering({"cancel_order": [dependency("list_orders", "cancel_order", "order_id")]}))

    graph = await ToolGraph.build(catalog, model=second, base=base)

    assert [focus_of(call["prompt"]) for call in second.calls_for("ToolDependencies")] == ["cancel_order"]
    assert [(found.source, found.target) for found in graph.edges] == [
        ("find_user", "list_orders"),
        ("list_orders", "cancel_order"),
    ]


async def test_probe_turns_a_proposal_into_a_posterior_from_two_runs():
    model = ScriptedModel(
        plan=lambda messages, tools: text_message("answered"),
        answers={
            **answering({"find_user": [dependency("find_user", "list_orders", "user_id", 0.6)]}),
            "ProbeVerdict": lambda prompt: ProbeVerdict(with_source=0.8, without_source=0.2),
        },
    )

    graph = await ToolGraph.build(shop_tools(), model=model, probe=True)

    [found] = graph.edges
    assert found.weight == pytest.approx(0.48 / 0.56)
    assert found.evidence[0].kind == "probe"
    assert found.evidence[0].detail["prior"] == 0.6
    assert sorted(sorted(call["tools"]) for call in model.stream_calls) == [
        ["find_user", "list_orders"],
        ["list_orders"],
    ]


async def test_rejects_two_tools_with_one_name():
    tools = [make_tool("find_user", "One.", {"email": "string"}), make_tool("find_user", "Two.", {"email": "string"})]

    with pytest.raises(ValueError, match="unique"):
        await ToolGraph.build(tools, model=ScriptedModel())


async def test_rejects_a_minimum_weight_outside_zero_to_one():
    with pytest.raises(ValueError, match="min_edge_weight"):
        await ToolGraph.build(shop_tools(), model=ScriptedModel(), min_edge_weight=1.5)
