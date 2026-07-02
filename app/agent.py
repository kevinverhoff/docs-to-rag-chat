"""
Step 8: ReAct agent for document library.

Uses a custom LangGraph graph that forces tool_choice="required" on the first
model call, so the agent cannot skip tools and answer from training memory.
After at least one tool result is available, subsequent calls use tool_choice="auto"
so the agent can synthesize a final answer.

Usage:
  from agent import Agent
  from rag_pipeline import RagPipeline
  pipeline = RagPipeline()
  ag = Agent(pipeline)
  result = ag.chat("What challenges have the Ekumen worlds reported?")
  print(result["answer"])
"""

import argparse
import sys
from pathlib import Path
from typing import Annotated, TypedDict

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import ToolNode

from rag_pipeline import RagPipeline
from tools import make_tools

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / "secrets" / ".env")

from providers import build_chat_model

THEMES_PATH   = PROJECT_ROOT / "data" / "themes.parquet"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.json"

SYSTEM_PROMPT = (PROJECT_ROOT / "prompts" / "agent_system_prompt.txt").read_text(encoding="utf-8")


# ------------------------------------------------------------------
# State + graph
# ------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _build_graph(model, tools: list):
    """
    ReAct graph with forced tool use on the first step.
    - First model call: tool_choice="required" -- cannot skip tools
    - Subsequent calls: tool_choice="auto"     -- can synthesize final answer
    """
    tool_node      = ToolNode(tools)
    model_required = model.bind_tools(tools, tool_choice="required")
    model_auto     = model.bind_tools(tools)

    def call_model(state: AgentState) -> dict:
        messages = state["messages"]
        has_tool_results = any(isinstance(m, ToolMessage) for m in messages)
        llm = model_auto if has_tool_results else model_required
        return {"messages": [llm.invoke(messages)]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


# ------------------------------------------------------------------
# Agent class
# ------------------------------------------------------------------

class Agent:
    """
    ReAct agent over the document library.
    Initialize once and share across conversations (@st.cache_resource).
    """

    def __init__(
        self,
        pipeline: RagPipeline,
        themes_path: Path = THEMES_PATH,
        metadata_path: Path = METADATA_PATH,
    ) -> None:
        import json as _json

        themes_df: pd.DataFrame | None = None
        if themes_path.exists():
            themes_df = pd.read_parquet(themes_path)

        metadata: list[dict] | None = None
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                metadata = _json.load(f)

        tools        = make_tools(pipeline, themes_df, metadata)
        model        = build_chat_model()
        self._model  = model
        self.app     = _build_graph(model, tools)

    def chat(
        self,
        question: str,
        *,
        history: list[dict] | None = None,
        tag_filters: "dict[str, str | None] | None" = None,
        theme_cluster: str | None = None,
    ) -> dict:
        """
        Run a single turn. Returns {"answer": str, "messages": list}.
        Pass history as [{"role": "user"/"assistant", "content": str}, ...].
        """
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        messages.extend(_history_to_messages(history))

        filter_lines = [
            f"{k.replace('_', ' ').title()}: {v}"
            for k, v in (tag_filters or {}).items()
            if v
        ]
        if theme_cluster:
            filter_lines.append(f"Theme cluster: {theme_cluster}")

        user_text = question
        if filter_lines:
            user_text = "Active filters -- " + " | ".join(filter_lines) + "\n\n" + question

        messages.append(HumanMessage(content=user_text))

        accumulated: list[BaseMessage] = list(messages)
        hit_limit = False

        try:
            for chunk in self.app.stream(
                {"messages": messages},
                config={"recursion_limit": 50},
            ):
                for node_output in chunk.values():
                    accumulated.extend(node_output.get("messages", []))
        except GraphRecursionError:
            hit_limit = True

        if hit_limit:
            final_msg = self._synthesize_partial(accumulated)
        else:
            final_msg = accumulated[-1]

        return {
            "answer":   final_msg.content,
            "messages": accumulated,
        }


    def _synthesize_partial(self, messages: list[BaseMessage]) -> AIMessage:
        """
        Called when the recursion limit is hit. Asks the model to summarize
        whatever tool results it gathered and prompt the user for a follow-up.
        """
        synthesis_prompt = (
            "You ran out of steps before finishing your research. "
            "Summarize the information you gathered so far into a helpful partial response. "
            "Be clear about what you found and what you didn't get to explore. "
            "End with a short note â€” something like: "
            "'This is a partial response. To go deeper, try asking a more specific "
            "follow-up question (e.g., about a single program, district, or document type).'"
        )
        synth_messages = list(messages) + [HumanMessage(content=synthesis_prompt)]
        return self._model.invoke(synth_messages)

def _history_to_messages(history: list[dict] | None) -> list:
    if not history:
        return []
    out = []
    for h in history:
        role    = h.get("role", "user")
        content = h.get("content", "")
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
    return out


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chat with the document agent.",
        epilog="""Examples:
  python agent.py "What are the key findings from the Hainish worlds?"
  python agent.py "Summarize document themes" --filter "program=Ekumen Outreach"
  python agent.py "What appears in the Gethen reports?" --filter "district=Gethen"
  python agent.py "Compare site visits" --filter "doc_type=site_visit" --filter "program=Ansible Studies"
""",
    )
    parser.add_argument("question")
    parser.add_argument(
        "--filter", action="append", default=[], metavar="KEY=VALUE",
        help="Tag filter, e.g. --filter 'program=Ekumen Outreach'",
    )
    parser.add_argument("--theme-cluster", default=None, dest="theme_cluster")
    args = parser.parse_args()

    pipeline = RagPipeline()
    ag = Agent(pipeline)

    tag_filters = {}
    for f in args.filter:
        if "=" in f:
            k, v = f.split("=", 1)
            tag_filters[k.strip()] = v.strip().strip("'\"")

    result = ag.chat(
        args.question,
        tag_filters=tag_filters or None,
        theme_cluster=args.theme_cluster,
    )
    print(result["answer"])
