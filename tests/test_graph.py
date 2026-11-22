import json

import pytest

from strands_tool_graph import ToolGraph
from tests.support import edge, shop_graph


def test_graph_round_trips_through_a_file(tmp_path):
    graph = shop_graph()
    path = tmp_path / "tool_graph.json"

    graph.save(path)
    loaded = ToolGraph.load(path)

    assert loaded == graph
    assert loaded.version == graph.version


def test_version_changes_when_an_edge_weight_changes():
    assert shop_graph(find_to_list=0.9).version != shop_graph(find_to_list=0.4).version


def test_edges_are_indexed_by_source_and_target():
    graph = shop_graph()

    assert [found.target for found in graph.outgoing("find_user")] == ["list_orders"]
    assert [found.source for found in graph.incoming("refund_order")] == ["list_orders"]
    assert graph.incoming("find_user") == ()


def test_rejects_edges_to_tools_outside_the_graph():
    graph = shop_graph()

    with pytest.raises(ValueError, match="delete_account"):
        ToolGraph(nodes=graph.nodes, edges=(edge("find_user", "delete_account", "user_id", 0.9),))


def test_rejects_an_unknown_file_format(tmp_path):
    path = tmp_path / "tool_graph.json"
    path.write_text(json.dumps({"format": "something-else", "nodes": [], "edges": []}))

    with pytest.raises(ValueError, match="unsupported tool graph format"):
        ToolGraph.load(path)
