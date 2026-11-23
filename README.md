# strands-tool-graph

ToolGraph for [Strands Agents](https://strandsagents.com). The package builds a dependency graph of your tools one time. Before each model call, it scores candidate next actions with the graph. Then it adds the scores to the prompt as advice. The agent still selects the next action.

The method comes from "ToolGraph as a Q-Function for Test-Time Scaling of Tool-Use Agents" (Shen and others, NeurIPS 2026 PTA workshop, [arXiv:2611.20569](https://arxiv.org/abs/2611.20569)). The authors did not release code. This package is an independent implementation from the paper.

## Install

The package is not on PyPI yet. Install it from GitHub:

```bash
pip install "git+https://github.com/arielnabavianpersonal/strands-tool-graph.git"
```

## Quick start

```python
from strands import Agent
from strands_tool_graph import ToolGraphAdvisor

agent = Agent(tools=[find_user, list_orders, refund_order], plugins=[ToolGraphAdvisor()])
agent("Refund my last order. My email is ana@example.com.")
```

On the first model call, the advisor builds the graph from the tools of the agent, with `agent.aux_model`. When the agent gets new or changed tools, the advisor updates the graph for those tools only.

To keep the graph between runs, give a file. The advisor reads the file when it starts, and writes the file after each update.

```python
advisor = ToolGraphAdvisor(graph_file="tool_graph.json")
```

## Build the graph ahead of time

You can also build the graph before the agent runs, for example in CI or to examine the edges first. `ToolGraph.build` uses a default Bedrock model. To use a different model, give `model=`.

```python
graph = await ToolGraph.build(tools)
graph.save("tool_graph.json")

agent = Agent(tools=tools, plugins=[ToolGraphAdvisor(graph)])
```

`advisor.graph` gives the graph that the advisor uses at this time.

## How the builder makes the graph

The graph has one node for each tool. An edge from tool A to tool B means that the output of A can fill a necessary input of B. Edges come from three sources of evidence:

1. **Schema match.** A declared output field of A has the same name and type as a necessary input of B. This source makes no model calls.
2. **Model proposals.** One model call for each tool reads the catalog. The model gives the dependencies of that tool in the two directions.
3. **Probes** (`probe=True`, off by default). For each proposed edge, the builder runs a short agent with A and B, then with B only. A judge compares the two attempts. The result changes the edge confidence, with the proposal as the prior.

The weight of an edge is the mean confidence of its evidence. The builder removes edges with a weight of `min_edge_weight` (default 0.3) or less. Each edge keeps its evidence in `edge.evidence`, so you can see why the edge exists.

> **CAUTION:** Probes run your real tools. Use `probe=True` only with tools that cannot change real data, for example sandbox tools or mocks.

The advisor updates its graph by itself. For a graph that you build ahead of time, give the old graph as `base` when the catalog changes. The builder then examines only new and changed tools, and removes the tools that are gone.

```python
graph = await ToolGraph.build(tools, base=ToolGraph.load("tool_graph.json"))
```

## How the advice works

`ToolGraphAdvisor` runs before each model call of the agent. For each call, it does these steps:

1. It finds up to three goal tools for the user request. It finds them one time for each request.
2. It samples `candidates` next actions from the agent's own model, with the same messages and tools.
3. It calculates a graph value for each candidate and goal (`Q_dp` in the paper). The value comes from value iteration over sets of available inputs.
4. It marks a candidate as not admissible when a necessary input is missing. It does the same when a tool must supply an input and the value does not occur in the conversation.
5. It scores each admissible candidate with the selected scorer.
6. It writes the top rows and the graph evidence as short advice, with one more model call.
7. It adds the advice to the last user message, for this call only. The advice does not go into `agent.messages`.

| `scorer` | Score | Extra model calls |
|---|---|---|
| `"llm"` (default) | A model reads the conversation, the graph evidence and the graph value. Then it gives a score from 0 to 10 (`Q_llm`). | `scorer_samples` for each admissible candidate |
| `"dp"` | The graph value (`Q_dp`). | None |
| `"pagerank"` | Personalized PageRank from the request over the graph. | None |

In the paper, only the `"llm"` scorer did better than PageRank on the same graph. Its mean success was 71.0, and the mean success of PageRank was 62.7.

## Settings

| Setting | Default | Effect |
|---|---|---|
| `graph` | `None` | Sets a graph to start from. The advisor still updates it for new tools. |
| `graph_file` | `None` | Sets a file that keeps the graph between runs. |
| `probe` | `False` | Lets the builder run probes with the real tools of the agent. |
| `scorer` | `"llm"` | Selects how the advisor ranks candidates. |
| `candidates` | 4 | Sets the number of sampled next actions for each model call. |
| `top_k` | 8 | Sets the number of action and goal rows in the advice. |
| `hops` | 1 | Sets the radius of the graph evidence around the candidate and goal tools. |
| `min_edge_weight` | 0.0 | Removes edges below this weight from the evidence. |
| `scorer_samples` | 1 | Sets the number of scorer calls for each candidate. The paper used 5 at temperature 1.0. |
| `model` | `None` | Sets the model for the graph, goals, scores and advice. `None` selects `agent.aux_model`. |

## Cost

Each model call of the agent adds these model calls:

- `candidates` calls to the agent's model, with the full conversation.
- `scorer_samples` calls for each admissible candidate, with the `"llm"` scorer only.
- One call for the advice.
- One call for the goals, for each new user request.

The first model call also builds the graph. The build makes one model call for each tool, and more calls for the probes when `probe=True`. After a change to the tools, the update makes calls for the new and changed tools only.

`advisor.usage` gives the tokens of these calls. The usage of the agent does not include them.

## Limits

- The advisor gives no advice to a model that keeps the conversation on the server.
- The graph value knows tool types, not argument values. Thus it cannot see a correct tool with a wrong argument. The `"llm"` scorer reads the arguments.
- The candidate samples and the scorer samples use the temperature of their model. To get different samples, set a higher temperature on that model.
- PageRank starts from the word overlap between the request and the tool descriptions. The paper used embeddings.
- The package does not include the GraphSAGE scorer (`Q_emb`) from the paper.

## Development

```bash
pip install -e ".[dev]"
hatch run prepare
```

## Release

The `publish.yml` workflow publishes the package to PyPI when you publish a GitHub release with a `v` tag, for example `v0.1.0`. Before the first release, configure PyPI trusted publishing with the environment name `pypi`.

## License

Apache 2.0. See [LICENSE](LICENSE).
