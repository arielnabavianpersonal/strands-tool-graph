import pytest

from strands_tool_graph import _values
from strands_tool_graph._state import Candidate, DecisionState, ToolCall
from strands_tool_graph._values import DependencyValues, ValueIterationError, best_path, neighborhood, pagerank_scores
from tests.support import shop_graph, tool_turn

ASK = {"role": "user", "content": [{"text": "Please refund my order. My email is ana@shop.test."}]}


def state(*turns):
    return DecisionState.from_messages([ASK, *[message for turn in turns for message in turn]])


def call(name, **arguments):
    return Candidate(calls=(ToolCall(name, arguments),))


def value(candidate, *turns, graph=None):
    return DependencyValues(graph or shop_graph(), state(*turns)).candidate_value(candidate, "refund_order")


def test_a_call_that_unlocks_the_goal_beats_an_unrelated_call():
    unlocking = value(call("find_user", email="ana@shop.test"))
    unrelated = value(call("get_weather", city="Paris"))

    assert unlocking > unrelated


def test_progress_beats_a_reply():
    progress = value(call("find_user", email="ana@shop.test"))
    reply = value(Candidate(calls=(), text="Which order?"))

    assert progress > reply > 0.5


def test_calling_the_goal_with_grounded_inputs_scores_one():
    turns = (
        tool_turn("find_user", {"email": "ana@shop.test"}, {"user_id": "u-1"}),
        tool_turn("list_orders", {"user_id": "u-1"}, {"order_id": "o-7"}),
    )

    assert value(call("refund_order", order_id="o-7", reason="damaged"), *turns) == 1.0


@pytest.mark.parametrize(
    "candidate",
    [
        call("list_orders", user_id="u-999"),
        call("find_user"),
        call("delete_account", user_id="u-1"),
    ],
    ids=["ungrounded producible input", "missing conversation input", "tool outside the graph"],
)
def test_candidates_without_their_inputs_are_not_admissible(candidate):
    assert value(candidate) is None


def test_an_input_found_in_the_conversation_makes_a_call_admissible():
    turns = (tool_turn("find_user", {"email": "ana@shop.test"}, {"user_id": "u-1"}),)

    assert value(call("list_orders", user_id="u-1"), *turns) is not None


def test_a_tool_that_only_failed_loses_value():
    failed = (tool_turn("find_user", {"email": "ana@shop.test"}, {"error": "timeout"}, status="error"),)

    assert value(call("find_user", email="ana@shop.test"), *failed) < value(call("find_user", email="ana@shop.test"))


def test_weak_edges_lower_the_value_of_a_path():
    strong = value(call("find_user", email="ana@shop.test"), graph=shop_graph(find_to_list=0.9, list_to_refund=0.9))
    weak = value(call("find_user", email="ana@shop.test"), graph=shop_graph(find_to_list=0.4, list_to_refund=0.4))

    assert strong > weak


def test_feasible_alternatives_are_executable_tools_on_a_path_to_the_goal():
    values = DependencyValues(shop_graph(), state(tool_turn("find_user", {"email": "a"}, {"user_id": "u-1"})))

    assert values.feasible(["refund_order"], limit=16) == ("find_user", "list_orders")


def test_raises_when_the_entity_lattice_is_too_large(monkeypatch):
    monkeypatch.setattr(_values, "_MAX_STATES", 1)

    with pytest.raises(ValueIterationError):
        value(call("find_user", email="ana@shop.test"))


def test_pagerank_spreads_request_relevance_along_dependency_edges():
    ranks = pagerank_scores(shop_graph(), "my email")

    assert ranks["find_user"] == 1.0
    assert ranks["list_orders"] > 0
    assert ranks["refund_order"] > 0
    assert ranks["get_weather"] == 0


def test_pagerank_falls_back_to_uniform_relevance_when_nothing_matches():
    ranks = pagerank_scores(shop_graph(), "xyzzy")

    assert all(rank > 0 for rank in ranks.values())


def test_best_path_multiplies_edge_weights():
    path, confidence = best_path(shop_graph(), "find_user", "refund_order", min_edge_weight=0.0)

    assert path == ("find_user", "list_orders", "refund_order")
    assert confidence == pytest.approx(0.72)
    assert best_path(shop_graph(), "find_user", "refund_order", min_edge_weight=0.85) is None


def test_neighborhood_grows_one_hop_at_a_time():
    graph = shop_graph()

    assert neighborhood(graph, ["refund_order"], hops=0, min_edge_weight=0.0) == {"refund_order"}
    assert neighborhood(graph, ["refund_order"], hops=1, min_edge_weight=0.0) == {"refund_order", "list_orders"}
    assert neighborhood(graph, ["refund_order"], hops=2, min_edge_weight=0.0) == {
        "refund_order",
        "list_orders",
        "find_user",
    }
