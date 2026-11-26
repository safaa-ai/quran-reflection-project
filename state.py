
from typing import Any, Dict, List, Optional, TypedDict


class Task(TypedDict, total=False):
    task_id: int
    action_text: str
    task: str
    status: str
    due_date: str
    date: str
    time: str
    duration_min: int
    custom_actions: str
    is_custom: bool


class QuranReflectionState(TypedDict, total=False):

    # Quran and RAG
    surah_number: int
    verse_number: int
    verse_text: str
    tafsir_text: str
    quran_context: str

    # Reflection
    reflection_text: str
    suggested_actions: List[str]
    reflection_warnings: List[str]

    # User selection
    selected_action: str
    custom_action: str
    custom_actions: List[str]
    is_custom: bool

    # User preferences
    user_id: str
    available_minutes: int
    start_date: str
    end_date: str
    duration_days: int
    preferred_time: str

    # Planning
    weekly_plan: List[Dict[str, Any]]
    plan_summary: str
    plan_warnings: List[str]

    # Replanning
    run_replanning: bool
    replan_custom_action: bool
    completed_tasks: List[str]
    overdue_tasks: List[Task]
    revised_plan: List[Dict[str, Any]]
    replanning_message: str

    # General
    messages: List[str]
    error: Optional[str]
