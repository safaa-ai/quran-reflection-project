
from typing import Any, Dict, List

from langgraph.graph import StateGraph, START, END

from state import QuranReflectionState
from planner_agent import generate_reflection, plan_action
from replanning_agent import (
    generate_replanning_plan_with_ai,
    generate_custom_task_replanning,
)
from rag_retriever import format_rag_prompt_context


# ==========================================
# 1. RAG NODE
# ==========================================

def rag_node(state: QuranReflectionState) -> dict:
    """Retrieve Quran verse and tafsir from the RAG database."""

    verse_num = state.get("verse_number")

    if verse_num is None:
        return {"error": "verse_number is required."}

    try:
        verse_num = int(verse_num)
    except (TypeError, ValueError):
        return {"error": "verse_number must be a valid integer."}

    context = format_rag_prompt_context(verse_num)

    if not context:
        return {
            "error": f"No Quran context found for verse {verse_num}."
        }

    return {
        "quran_context": context,
        "tafsir_text": context,
        "error": None,
    }


# ==========================================
# 2. ROUTE AFTER RAG
# ==========================================

def route_after_rag(state: QuranReflectionState) -> str:
    """Choose the correct workflow path."""

    if state.get("error"):
        return END

    # Replanning has priority over creating a new plan.
    if state.get("run_replanning") and (
        state.get("replan_custom_action")
        or state.get("overdue_tasks")
    ):
        return "replanning"

    # Normal planning.
    if state.get("selected_action") or state.get("custom_action"):
        return "planner"

    # Reflection only.
    return "reflection"


# ==========================================
# 3. REFLECTION NODE
# ==========================================

def reflection_node(state: QuranReflectionState) -> dict:
    """Generate reflection and suggested behavioral actions."""

    context = state.get("quran_context") or ""

    if not context:
        return {"error": "Quran context is missing."}

    try:
        result = generate_reflection(quran_context=context)

        return {
            "reflection_text": result.get("reflection_text", ""),
            "suggested_actions": result.get("suggested_actions", []),
            "reflection_warnings": result.get("warnings", []),
            "error": None,
        }

    except Exception as e:
        return {"error": f"Reflection error: {str(e)}"}


# ==========================================
# 4. PLANNER NODE
# ==========================================

def planner_node(state: QuranReflectionState) -> dict:
    """Create a schedule for the selected or custom action."""

    action_text = (
        state.get("custom_action")
        or state.get("selected_action")
        or ""
    ).strip()

    if not action_text:
        return {
            "error": "Please select an action before planning."
        }

    context = state.get("quran_context") or ""

    try:
        result = plan_action(
            action_text=action_text,
            available_minutes=int(
                state.get("available_minutes") or 30
            ),
            quran_context=context,
            start_date=state.get("start_date"),
            end_date=state.get("end_date"),
            duration_days=state.get("duration_days"),
            preferred_time=state.get("preferred_time", "any"),
            is_custom=bool(
                state.get("is_custom") or state.get("custom_action")
            ),
        )

        return {
            "weekly_plan": result.get("tasks", []),
            "plan_summary": result.get("summary", ""),
            "plan_warnings": result.get("warnings", []),
            "start_date": result.get("start_date", ""),
            "end_date": result.get("end_date", ""),
            "error": None,
        }

    except Exception as e:
        return {"error": f"Planner error: {str(e)}"}


# ==========================================
# 5. PREPARE OVERDUE TASKS
# ==========================================

def prepare_overdue_tasks(
    tasks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Normalize tasks before sending them to the replanning agent."""

    prepared_tasks = []

    for index, task in enumerate(tasks, start=1):

        prepared_tasks.append({
            "task_id": task.get("task_id", index),
            "action_text": (
                task.get("action_text")
                or task.get("task", "")
            ),
            "status": task.get("status", "pending"),
            "due_date": (
                task.get("due_date")
                or task.get("date", "")
            ),
        })

    return prepared_tasks


# ==========================================
# 6. REPLANNING NODE
# ==========================================

def replanning_node(state: QuranReflectionState) -> dict:
    """Handle custom actions or overdue tasks."""

    try:

        # ----------------------------------
        # Custom action replanning
        # ----------------------------------

        if state.get("replan_custom_action"):

            action_text = (
                state.get("custom_action")
                or state.get("selected_action")
                or ""
            ).strip()

            if not action_text:
                return {
                    "error": "Custom action is required."
                }

            start_date = state.get("start_date")
            end_date = state.get("end_date")

            if not start_date or not end_date:
                return {
                    "error": "Start date and end date are required."
                }

            # Use the actual previous due date if supplied.
            # Do not invent a previous due date.
            previous_due_date = (
                state.get("previous_due_date")
                or state.get("due_date")
                or ""
            )

            result = generate_custom_task_replanning(
                task={
                    "task_id": 1,
                    "action_text": action_text,
                    "custom_actions": action_text,
                    "is_custom": True,
                    "status": "pending",
                    "due_date": previous_due_date,
                },
                start_date=str(start_date),
                end_date=str(end_date),
                available_minutes=state.get("available_minutes"),
            )

            suggestion = result.get("suggestion", "")
            encouragement = result.get("encouragement", "")

            replanning_message = "\n\n".join(
                part for part in [suggestion, encouragement] if part
            )

            return {
                "revised_plan": [result],
                "replanning_message": replanning_message,
                "error": None,
            }

        # ----------------------------------
        # Existing overdue tasks replanning
        # ----------------------------------

        tasks = prepare_overdue_tasks(
            state.get("overdue_tasks", [])
        )

        if not tasks:
            return {
                "revised_plan": [],
                "replanning_message": "لا توجد مهام لإعادة تخطيطها.",
                "error": None,
            }

        result = generate_replanning_plan_with_ai(
            tasks=tasks,
            available_minutes=state.get("available_minutes"),
        )

        return {
            "revised_plan": result.get("recommendations", []),
            "replanning_message": result.get("encouragement", ""),
            "error": None,
        }

    except Exception as e:
        return {"error": f"Replanning error: {str(e)}"}


# ==========================================
# 7. BUILD LANGGRAPH WORKFLOW
# ==========================================

graph = StateGraph(QuranReflectionState)

graph.add_node("rag", rag_node)
graph.add_node("reflection", reflection_node)
graph.add_node("planner", planner_node)
graph.add_node("replanning", replanning_node)

graph.add_edge(START, "rag")

graph.add_conditional_edges(
    "rag",
    route_after_rag,
    {
        "reflection": "reflection",
        "planner": "planner",
        "replanning": "replanning",
        END: END,
    },
)

graph.add_edge("reflection", END)
graph.add_edge("planner", END)
graph.add_edge("replanning", END)

app = graph.compile()
