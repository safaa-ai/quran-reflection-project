from typing import TypedDict, List, Optional, Dict, Any


class QuranReflectionState(TypedDict, total=False):
    verse_number: int
    verse_text: str
    tafsir_text: str
    suggested_actions: List[str]
    selected_action: str
    user_id: str
    available_minutes: int
    weekly_plan: List[Dict[str, Any]]
    completed_tasks: List[str]
    overdue_tasks: List[str]
    revised_plan: List[Dict[str, Any]]
    replanning_message: str
    messages: List[str]
    error: Optional[str]