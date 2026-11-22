from strands_tool_graph._state import Candidate, DecisionState, ToolCall
from tests.support import call_message, tool_turn

ASK = {"role": "user", "content": [{"text": "Refund order o-7."}]}
FOLLOW_UP = {"role": "user", "content": [{"text": "The reason is damage."}]}


def test_counts_successes_and_failures_per_tool():
    state = DecisionState.from_messages(
        [
            ASK,
            *tool_turn("list_orders", {"user_id": "u-1"}, {"error": "timeout"}, status="error"),
            *tool_turn("list_orders", {"user_id": "u-1"}, {"order_id": "o-7"}),
            *tool_turn("refund_order", {"order_id": "o-7"}, {"error": "denied"}, status="error"),
        ]
    )

    assert state.succeeded == {"list_orders"}
    assert state.failed_only == {"refund_order"}
    assert state.history("list_orders") == "called 2x (1 ok, 1 failed)"
    assert state.history("find_user") == "not called yet"


def test_request_joins_user_asks_and_skips_tool_results():
    state = DecisionState.from_messages(
        [ASK, *tool_turn("list_orders", {"user_id": "u-1"}, {"order_id": "o-7"}), FOLLOW_UP]
    )

    assert state.request == "Refund order o-7.\nThe reason is damage."


def test_grounds_values_that_appear_in_the_conversation():
    state = DecisionState.from_messages([ASK, *tool_turn("find_user", {"email": "a@b.co"}, {"user_id": "U-42"})])

    assert state.grounds("o-7")
    assert state.grounds("u-42")
    assert not state.grounds("o-8")
    assert not state.grounds("")


def test_short_values_must_match_a_whole_token():
    state = DecisionState.from_messages([{"role": "user", "content": [{"text": "Seat 12A please, row 7."}]}])

    assert state.grounds("7")
    assert not state.grounds("2")


def test_candidate_reads_tool_calls_and_text_from_a_message():
    message = call_message(("find_user", {"email": "a@b.co"}), ("get_weather", {"city": "Oslo"}))
    message["content"].append({"text": "Checking."})

    candidate = Candidate.from_message(message)

    assert candidate.tools == ("find_user", "get_weather")
    assert candidate.text == "Checking."
    assert candidate.describe() == 'find_user(email="a@b.co") then get_weather(city="Oslo")'


def test_candidates_with_the_same_calls_share_a_key():
    first = Candidate(calls=(ToolCall("find_user", {"email": "a@b.co"}),), text="one")
    second = Candidate(calls=(ToolCall("find_user", {"email": "a@b.co"}),), text="two")

    assert first.key == second.key
    assert Candidate(calls=()).key == "reply"
