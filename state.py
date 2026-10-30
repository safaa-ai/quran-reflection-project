from typing import TypedDict, List, Optional, Dict, Any
class Task(TypedDict):
    task_id: int
    action_text: str
    status: str
    due_date: str


class QuranReflectionState(TypedDict, total=False):
    verse_number: int
    verse_text: str
    tafsir_text: str
    suggested_actions: List[str]
    selected_action: str
    user_id: str
    available_minutes: int
    start_date: str
    preferred_time: str
    
    weekly_plan: List[Dict[str, Any]]
    completed_tasks: List[str]
    overdue_tasks: List[Task]
    revised_plan: List[Dict[str, Any]]
    replanning_message: str
    messages: List[str]
    error: Optional[str]
