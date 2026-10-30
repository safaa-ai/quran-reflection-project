from typing import Any
from langgraph.graph import StateGraph, START, END

from state import QuranReflectionState
from planner_agent import plan_action
from replanning_agent import generate_replanning_plan_with_ai


def planner_node(state: QuranReflectionState) -> dict:
    result = plan_action(
        action_text=state.get("selected_action", ""),
        available_minutes=state.get("available_minutes", 30),
        quran_context=state.get("tafsir_text", ""),
        start_date=state.get("start_date"),
        preferred_time=state.get("preferred_time", "any"),
    )

    return {
        "weekly_plan": result.get("tasks", []),
        "error": None
    }


def replanning_node(state: QuranReflectionState) -> dict:
    tasks = state.get("overdue_tasks", [])

    result = generate_replanning_plan_with_ai(
        tasks=tasks,
        available_minutes=state.get("available_minutes")
    )

    return {
        "revised_plan": result.get("recommendations", []),
        "replanning_message": result.get("encouragement", "")
    }


def check_overdue_tasks(state: QuranReflectionState):
    if state.get("overdue_tasks"):
        return "replanning"
    return END


graph = StateGraph(QuranReflectionState)

graph.add_node("planner", planner_node)
graph.add_node("replanning", replanning_node)

graph.add_edge(START, "planner")

graph.add_conditional_edges(
    "planner",
    check_overdue_tasks,
    {
        "replanning": "replanning",
        END: END
    }
)

graph.add_edge("replanning", END)

app = graph.compile()
