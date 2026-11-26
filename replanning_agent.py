"""
Adaptive Replanning Agent — Week 2
----------------------------------
Keeps the existing rule-based replanning logic and adds an optional Gemini-powered
Arabic suggestion for unfinished tasks delayed by more than 48 hours.

Expected task format:
{
    "task_id": 1,
    "action_text": "كتابة فائدة تدبرية",
    "status": "pending",  # e.g. pending / completed
    "due_date": "2026-10-22"  # YYYY-MM-DD
}

Setup for Gemini:
1. Install dependencies:
   pip install langchain-google-genai python-dotenv
2. Create a .env file beside this file and add:
   GOOGLE_API_KEY=your_google_gemini_api_key
3. Run this file with Python.

Security: never commit the .env file or share your API key.
"""

from datetime import datetime
from typing import TypedDict, List, Dict, Any, Optional
import os


# Load .env if python-dotenv is installed. Rule-based replanning still works without it.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Shape of each task expected by this agent.
class Task(TypedDict, total=False):
    task_id: int
    action_text: str
    status: str
    due_date: str  # YYYY-MM-DD


def calculate_delay_hours(
    due_date_str: str,
    current_time: Optional[datetime] = None,
) -> float:
    """
    Calculate how many hours have passed since the task's due date.

    Since the existing data model stores only YYYY-MM-DD, the due date is
    interpreted as midnight at the start of that date (local time).
    Returns 0 for missing/invalid dates or dates that are not overdue.
    """
    if not due_date_str:
        return 0.0

    try:
        due_date = datetime.strptime(due_date_str, "%Y-%m-%d")
    except (TypeError, ValueError):
        return 0.0

    now = current_time or datetime.now()
    if now > due_date:
        return (now - due_date).total_seconds() / 3600
    return 0.0


def calculate_task_statistics(
    tasks: List[Task],
    current_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Calculate task counts, completion/delay rates, and tasks late over 48 hours."""
    total_tasks = len(tasks)

    if total_tasks == 0:
        return {
            "total_tasks": 0,
            "completed_tasks": 0,
            "delayed_tasks": 0,
            "completion_rate": 0.0,
            "delay_rate": 0.0,
            "late_over_48h": [],
        }

    delayed_tasks: List[Dict[str, Any]] = []
    late_over_48h: List[Dict[str, Any]] = []
    completed_tasks = [task for task in tasks if task["status"].lower() == "completed"]

    for task in tasks:
        if task["status"].lower() == "completed":
            continue

        delay_hours = calculate_delay_hours(task.get("due_date", ""), current_time)
        if delay_hours > 0:
            delayed_task = {**task, "delay_hours": round(delay_hours, 2)}
            delayed_tasks.append(delayed_task)
            if delay_hours > 48:
                late_over_48h.append(delayed_task)

    return {
        "total_tasks": total_tasks,
        "completed_tasks": len(completed_tasks),
        "delayed_tasks": len(delayed_tasks),
        "completion_rate": round(len(completed_tasks) / total_tasks * 100, 2),
        "delay_rate": round(len(delayed_tasks) / total_tasks * 100, 2),
        "late_over_48h": late_over_48h,
    }


def generate_replanning_plan(
    tasks: List[Task],
    current_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Generate the original rule-based replanning plan.
    Kept available so existing code (including an agent graph) can keep calling it.
    """
    statistics = calculate_task_statistics(tasks, current_time)
    recommendations: List[Dict[str, Any]] = []

    for task in tasks:
        if task["status"].lower() == "completed":
            continue

        delay_hours = calculate_delay_hours(task.get("due_date", ""), current_time)

        if delay_hours > 48:
            recommendations.append({
                "task_id": task["task_id"],
                "action": "reschedule_and_split",
                "delay_hours": round(delay_hours, 2),
                "message": (
                    "قسّمي المهمة إلى خطوات أصغر، وحددي لها موعدًا جديدًا مناسبًا."
                ),
            })
        elif delay_hours > 0:
            recommendations.append({
                "task_id": task["task_id"],
                "action": "reschedule",
                "delay_hours": round(delay_hours, 2),
                "message": "حددي وقتًا جديدًا قريبًا لإنجاز المهمة.",
            })
        elif task["status"].lower() == "pending":
            recommendations.append({
                "task_id": task["task_id"],
                "action": "keep_scheduled",
                "delay_hours": 0.0,
                "message": (
                    "حافظي على موعد المهمة، ووزّعي وقتك عليها حسب أولوياتك."
                ),
            })

    if statistics["delayed_tasks"] > 0:
        encouragement = (
            "التأخير قابل للتدارك. ابدئي بخطوة صغيرة "
            "وعدّلي الجدول بما يناسب وقتك."
        )
    else:
        encouragement = "أحسنتِ! لا توجد مهام متأخرة. استمري على نفس الوتيرة."

    return {
        "statistics": statistics,
        "recommendations": recommendations,
        "encouragement": encouragement,
    }


REPLANNING_PROMPT = """
أنتِ مساعدة ذكية وداعمة لإعادة تخطيط المهام الأسبوعية ضمن تطبيق للتدبر القرآني.
المطلوب اقتراح خطة عملية باللغة العربية لمهمة متأخرة أكثر من 48 ساعة.

راعي القواعد التالية:
- استخدمي نبرة مشجعة ورحيمة وغير حُكمية، وتجنبي إشعار المستخدم بالذنب.
- اقترحي خطوات صغيرة وواضحة يمكن تنفيذها.
- لا تفترضي أن لدى المستخدم وقتًا غير مذكور، ولا تخترعي موعدًا أو تاريخًا جديدًا.
- إذا كان الوقت المتاح محدودًا، اقترحي البدء بخطوة قصيرة وتقسيم المهمة.
- حافظي على طبيعة المهمة ومعناها، خصوصًا إذا كانت مرتبطة بالقرآن أو التدبر.
- أعيدي الإجابة بالعربية في شكل: سبب مختصر لإعادة التخطيط، خطوات مقترحة، واقتراح عام لتحديد موعد جديد دون اختراع تاريخ محدد.
"""


def generate_ai_replanning_suggestion(
    task: Task,
    available_minutes: Optional[int] = None,
    current_time: Optional[datetime] = None,
    model_name: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    """
    Use Gemini to generate a tailored Arabic replanning suggestion.

    This function intentionally calls Gemini only for unfinished tasks delayed
    by MORE THAN 48 hours. For all other tasks, it returns a skipped result.
    """
    now = current_time or datetime.now()
    delay_hours = calculate_delay_hours(task.get("due_date", ""), now)

    if task["status"].lower() == "completed":
        return {
            "task_id": task["task_id"],
            "used_ai": False,
            "reason": "completed_task",
            "delay_hours": round(delay_hours, 2),
            "suggestion": "المهمة مكتملة بالفعل، فلا حاجة لإعادة تخطيطها.",
        }

    if delay_hours <= 48:
        return {
            "task_id": task["task_id"],
            "used_ai": False,
            "reason": "delay_not_over_48_hours",
            "delay_hours": round(delay_hours, 2),
            "suggestion": None,
        }

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY غير موجود. أضيفي المفتاح إلى ملف .env "
            "أو إلى متغيرات البيئة لتفعيل اقتراحات Gemini."
        )

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise RuntimeError(
            "مكتبة langchain-google-genai غير مثبتة. "
            "ثبّتيها بالأمر: pip install langchain-google-genai"
        ) from exc

    time_context = (
        f"الوقت المتاح للمهمة: {available_minutes} دقيقة."
        if available_minutes is not None
        else "الوقت المتاح غير محدد؛ لا تفترضي عدد دقائق معينًا."
    )

    prompt = f"""
{REPLANNING_PROMPT}

بيانات المهمة:
- رقم المهمة: {task['task_id']}
- وصف المهمة: {task['action_text']}
- الحالة: {task['status']}
- تاريخ الاستحقاق: {task.get('due_date', 'غير محدد')}
- عدد ساعات التأخير: {delay_hours:.2f}
- التاريخ والوقت الحاليان: {now.strftime('%Y-%m-%d %H:%M')}
- {time_context}

اكتبي اقتراحًا مخصصًا لهذه المهمة فقط.
"""

    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.4,
        )
        response = llm.invoke(prompt)
        content = response.content

        # Handle either plain text or a list of content blocks.
        if isinstance(content, list):
            suggestion = "\n".join(
                str(block.get("text", block)) if isinstance(block, dict) else str(block)
                for block in content
            ).strip()
        else:
            suggestion = str(content).strip()

        if not suggestion:
            raise RuntimeError("Gemini أعاد اقتراحًا فارغًا.")

        return {
            "task_id": task["task_id"],
            "used_ai": True,
            "reason": "delay_over_48_hours",
            "delay_hours": round(delay_hours, 2),
            "model": model_name,
            "suggestion": suggestion,
        }
    except Exception as exc:
        # Let the caller decide how to handle the API error; do not pretend AI succeeded.
        raise RuntimeError(f"فشل الحصول على اقتراح من Gemini: {exc}") from exc


def generate_replanning_plan_with_ai(
    tasks: List[Task],
    available_minutes: Optional[int] = None,
    current_time: Optional[datetime] = None,
    model_name: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    """
    Combined entry point:
    1. Creates the original rule-based plan.
    2. Replaces the generic message with a Gemini suggestion for each unfinished
       task delayed by more than 48 hours.
    3. Preserves the original statistics and recommendation structure.
    """
    plan = generate_replanning_plan(tasks, current_time)
    ai_suggestions: List[Dict[str, Any]] = []
    tasks_by_id = {task["task_id"]: task for task in tasks}

    for recommendation in plan["recommendations"]:
        if recommendation["action"] != "reschedule_and_split":
            continue

        task = tasks_by_id.get(recommendation["task_id"])
        if task is None:
            continue

        try:
            ai_result = generate_ai_replanning_suggestion(
                task=task,
                available_minutes=available_minutes,
                current_time=current_time,
                model_name=model_name,
            )
            ai_suggestions.append(ai_result)
            if ai_result.get("used_ai") and ai_result.get("suggestion"):
                recommendation["message"] = ai_result["suggestion"]
                recommendation["ai_generated"] = True
            else:
                recommendation["ai_generated"] = False
        except RuntimeError as exc:
            # Keep the safe rule-based recommendation if API setup/request fails.
            recommendation["ai_generated"] = False
            recommendation["ai_error"] = str(exc)
            ai_suggestions.append({
                "task_id": task["task_id"],
                "used_ai": False,
                "reason": "ai_error",
                "delay_hours": recommendation["delay_hours"],
                "error": str(exc),
            })

    plan["ai_suggestions"] = ai_suggestions
    return plan


# Week 3 additions — custom actions and date-aware replanning
def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def calculate_schedule_date_difference(
    old_date: Optional[str], proposed_date: Optional[str]
) -> Dict[str, Any]:
    """Return a transparent day difference between old and proposed dates."""
    old_dt, new_dt = _parse_date(old_date), _parse_date(proposed_date)
    if not old_dt or not new_dt:
        return {"days_difference": None, "direction": "unknown"}
    days = (new_dt.date() - old_dt.date()).days
    return {
        "days_difference": days,
        "direction": "later" if days > 0 else ("earlier" if days < 0 else "same_day"),
    }


def generate_custom_task_replanning(
    task: Task,
    start_date: str,
    end_date: str,
    available_minutes: Optional[int] = None,
    current_time: Optional[datetime] = None,
    model_name: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    """
    Replan a user-entered action within a chosen date range.
    Accepts custom_actions/action_text, reports date differences, and asks Gemini
    for supportive, practical adjustments without inventing specific dates.
    """
    start_dt, end_dt = _parse_date(start_date), _parse_date(end_date)
    if not start_dt or not end_dt or end_dt < start_dt:
        raise ValueError("يجب إدخال تاريخ بداية ونهاية صحيحين، والنهاية ليست قبل البداية.")

    action_text = task.get("action_text") or ""
    if not action_text.strip():
        raise ValueError("وصف العمل المخصص فارغ.")

    original_due = task.get("due_date")
    delay_hours = calculate_delay_hours(original_due, current_time)
    day_span = (end_dt.date() - start_dt.date()).days + 1
    minutes_context = (
        f"الوقت المتاح: {available_minutes} دقيقة."
        if available_minutes is not None
        else "الوقت المتاح غير محدد."
    )
    prompt = f"""
أنتِ مساعدة لإعادة تخطيط عمل سلوكي/تدبري اختاره المستخدم بنفسه.
أعيدي اقتراحه بأسلوب عربي دافئ، مشجع، غير حُكمي، عملي ومناسب للواقع.
لا تغيّري معنى العمل ولا تفترضي وقتًا غير مذكور، ولا تختلقي تواريخ.
قسّمي العمل إلى خطوات صغيرة، واقترحي توزيعًا مرنًا داخل نطاق التواريخ المحدد.
وضّحي بلطف كيف يمكن تعديل الموعد دون إشعار المستخدم بالذنب.
{REPLANNING_PROMPT}

العمل المختار (قد يكون مخصصًا): {action_text}
تاريخ الاستحقاق السابق: {original_due or 'غير محدد'}
التأخير بالساعات: {delay_hours:.2f}
النطاق الجديد: من {start_date} إلى {end_date} (مدته {day_span} يومًا).
{minutes_context}
اكتبي سببًا مختصرًا، خطوات قابلة للتنفيذ، واقتراح توزيع مرن داخل النطاق.
"""
    suggestion = None
    used_ai = False
    ai_error = None
    if os.getenv("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.4)
            response = llm.invoke(prompt)
            content = response.content
            if isinstance(content, list):
                suggestion = "\n".join(
                    str(block.get("text", block)) if isinstance(block, dict) else str(block)
                    for block in content
                ).strip()
            else:
                suggestion = str(content).strip()
            used_ai = bool(suggestion)
        except Exception as exc:
            ai_error = str(exc)
    else:
        ai_error = "GOOGLE_API_KEY غير متوفر؛ تم إعداد توصية احتياطية."

    if not suggestion:
        suggestion = (
            f"ابدئي بخطوة صغيرة من العمل «{action_text}»، ووزّعيها بمرونة "
            f"بين {start_date} و{end_date}. إن تعذّر يومٌ ما، انقلي الخطوة إلى "
            "وقت مناسب دون ضغط، وحافظي على الاستمرارية."
        )

    date_diff = calculate_schedule_date_difference(original_due, start_date)
    return {
        "task_id": task.get("task_id"),
        "action_text": action_text,
        "custom_task": bool(task.get("custom_actions") or task.get("is_custom")),
        "used_ai": used_ai,
        "ai_error": ai_error,
        "delay_hours": round(delay_hours, 2),
        "previous_due_date": original_due,
        "proposed_start_date": start_date,
        "proposed_end_date": end_date,
        "date_difference_from_previous_due_to_start": date_diff,
        "suggestion": suggestion,
        "encouragement": "التعديل فرصة للعودة بهدوء؛ خطوة صغيرة ثابتة خير من ضغط كبير.",
    }


if __name__ == "__main__":
    # Sample dates are intentionally in the past relative to 2026-10-28.
    sample_tasks: List[Task] = [
        {
            "task_id": 1,
            "action_text": "قراءة ورد من القرآن",
            "status": "completed",
            "due_date": "2026-10-20",
        },
        {
            "task_id": 2,
            "action_text": "كتابة فائدة تدبرية",
            "status": "pending",
            "due_date": "2026-10-22",
        },
        {
            "task_id": 3,
            "action_text": "تطبيق واجب سلوكي",
            "status": "pending",
            "due_date": "2026-10-19",
        },
        {
            "task_id": 4,
            "action_text": "مراجعة التفسير",
            "status": "pending",
            "due_date": "2026-10-25",
        },
    ]

    # The rule-based version always runs. The AI-enhanced version uses Gemini
    # for tasks delayed over 48 hours and falls back to rule-based messages on error.
    result = generate_replanning_plan_with_ai(
        sample_tasks,
        available_minutes=15,
    )

    # Example of replanning a custom action within a chosen date range.
    custom_result = generate_custom_task_replanning(
        task={
            "task_id": 99,
            "action_text": "التواصل مع شخص وصلة الرحم",
            "custom_actions": "التواصل مع شخص وصلة الرحم",
            "status": "pending",
            "due_date": "2026-10-22",
        },
        start_date="2026-11-27",
        end_date="2026-12-03",
        available_minutes=15,
    )
    print("=== Custom Task Replanning ===")
    print(custom_result)

    print("=== Task Statistics ===")
    print(result["statistics"])

    print("\n=== Replanning Recommendations ===")
    for recommendation in result["recommendations"]:
        print(recommendation)

    print("\n=== AI Suggestions / Status ===")
    for suggestion in result["ai_suggestions"]:
        print(suggestion)

    print("\n=== Encouragement ===")
    print(result["encouragement"])
