# !pip install langchain-google-genai python-dotenv

# ==========================================

print("kernel is working!")

# ==========================================

import os
from getpass import getpass
os.environ["GOOGLE_API_KEY"] = getpass("Enter your Gemini API key: ")
print("API key loaded:", bool(os.environ.get("GOOGLE_API_KEY")))

# ==========================================

import os
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
model="gemini-3.8-flash",
google_api_key=os.environ["GOOGLE_API_KEY"],
temperature=0.4
)

print("Gemini is ready!")

# ==========================================

response = llm.invoke("Say hello in Arabic")
print(response.content)

# ==========================================

text = response.content
if isinstance(text, list):
 for item in text:
  if isinstance(item, dict) and item.get("type") == "text":
   print(item.get("text", ""))
  else:
   print(text)

# ==========================================

from datetime import datetime 
from typing import TypedDict, List, Dict, Any, Optional
import os

# ==========================================


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class Task(TypedDict):
    task_id: int
    action_text: str
    status: str
    due_date: str


def calculate_delay_hours(
    due_date_str: str,
    current_time: Optional[datetime] = None,
) -> float:
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

    completed_tasks = [
        task for task in tasks
        if task["status"].lower() == "completed"
    ]

    delayed_tasks = []
    late_over_48h = []

    for task in tasks:
        if task["status"].lower() == "completed":
            continue

        delay_hours = calculate_delay_hours(
            task.get("due_date", ""),
            current_time,
        )

        if delay_hours > 0:
            delayed_task = {
                **task,
                "delay_hours": round(delay_hours, 2),
            }
            delayed_tasks.append(delayed_task)

            if delay_hours > 48:
                late_over_48h.append(delayed_task)

    return {
        "total_tasks": total_tasks,
        "completed_tasks": len(completed_tasks),
        "delayed_tasks": len(delayed_tasks),
        "completion_rate": round(
            len(completed_tasks) / total_tasks * 100, 2
        ),
        "delay_rate": round(
            len(delayed_tasks) / total_tasks * 100, 2
        ),
        "late_over_48h": late_over_48h,
    }


def generate_replanning_plan(
    tasks: List[Task],
    current_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    statistics = calculate_task_statistics(tasks, current_time)
    recommendations = []

    for task in tasks:
        if task["status"].lower() == "completed":
            continue

        delay_hours = calculate_delay_hours(
            task.get("due_date", ""),
            current_time,
        )

        if delay_hours > 48:
            recommendations.append({
                "task_id": task["task_id"],
                "action": "reschedule_and_split",
                "delay_hours": round(delay_hours, 2),
                "message": (
                    "قسّمي المهمة إلى خطوات أصغر، "
                    "وحددي لها موعدًا جديدًا مناسبًا."
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
                    "حافظي على موعد المهمة، "
                    "ووزّعي وقتك عليها حسب أولوياتك."
                ),
            })

    if statistics["delayed_tasks"] > 0:
        encouragement = (
            "التأخير قابل للتدارك. ابدئي بخطوة صغيرة "
            "وعدّلي الجدول بما يناسب وقتك."
        )
    else:
        encouragement = (
            "أحسنتِ! لا توجد مهام متأخرة. "
            "استمري على نفس الوتيرة."
        )

    return {
        "statistics": statistics,
        "recommendations": recommendations,
        "encouragement": encouragement,
    }


REPLANNING_PROMPT = """
أنتِ مساعدة ذكية وداعمة لإعادة تخطيط المهام الأسبوعية
ضمن تطبيق للتدبر القرآني.

أنشئي اقتراحًا عمليًا باللغة العربية لمهمة متأخرة أكثر من 48 ساعة.

القواعد:
- استخدمي نبرة مشجعة ورحيمة وغير حُكمية.
- تجنبي إشعار المستخدم بالذنب.
- اقترحي خطوات صغيرة وواضحة وقابلة للتنفيذ.
- لا تفترضي وقتًا متاحًا غير مذكور.
- لا تخترعي تاريخًا أو موعدًا محددًا.
- حافظي على طبيعة المهمة ومعناها.
- اقترحي تقسيم المهمة إذا كانت كبيرة.
- اكتبي سبب إعادة التخطيط، والخطوات المقترحة،
  وكيفية اختيار موعد جديد مناسب.
"""


def generate_ai_replanning_suggestion(
    task: Task,
    available_minutes: Optional[int] = None,
    current_time: Optional[datetime] = None,
    model_name: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    now = current_time or datetime.now()

    delay_hours = calculate_delay_hours(
        task.get("due_date", ""),
        now,
    )

    if task["status"].lower() == "completed":
        return {
            "task_id": task["task_id"],
            "used_ai": False,
            "reason": "completed_task",
            "delay_hours": round(delay_hours, 2),
            "suggestion": "المهمة مكتملة بالفعل.",
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
            "GOOGLE_API_KEY غير موجود. "
            "أضيفي مفتاح Gemini إلى متغيرات البيئة."
        )

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise RuntimeError(
            "ثبّتي المكتبات باستخدام: "
            "%pip install langchain-google-genai python-dotenv"
        ) from exc

    if available_minutes is None:
        time_context = "الوقت المتاح غير محدد."
    else:
        time_context = (
            f"الوقت المتاح للمهمة: {available_minutes} دقيقة."
        )

    prompt = f"""
{REPLANNING_PROMPT}

بيانات المهمة:
- رقم المهمة: {task['task_id']}
- وصف المهمة: {task['action_text']}
- الحالة: {task['status']}
- تاريخ الاستحقاق: {task.get('due_date', 'غير محدد')}
- التأخير بالساعات: {delay_hours:.2f}
- الوقت الحالي: {now.strftime('%Y-%m-%d %H:%M')}
- {time_context}

قدمي اقتراحًا مخصصًا لهذه المهمة باللغة العربية.
"""

    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.4,
        )

        response = llm.invoke(prompt)
        content = response.content

        if isinstance(content, list):
            suggestion = "\n".join(
                str(block.get("text", block))
                if isinstance(block, dict)
                else str(block)
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
        raise RuntimeError(
            f"فشل الحصول على اقتراح من Gemini: {exc}"
        ) from exc


def generate_replanning_plan_with_ai(
    tasks: List[Task],
    available_minutes: Optional[int] = None,
    current_time: Optional[datetime] = None,
    model_name: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    plan = generate_replanning_plan(tasks, current_time)

    ai_suggestions = []
    tasks_by_id = {
        task["task_id"]: task
        for task in tasks
    }

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

            if (
                ai_result.get("used_ai")
                and ai_result.get("suggestion")
            ):
                recommendation["message"] = ai_result["suggestion"]
                recommendation["ai_generated"] = True
            else:
                recommendation["ai_generated"] = False

        except RuntimeError as exc:
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


if __name__ == "__main__":
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

    result = generate_replanning_plan_with_ai(
        sample_tasks,
        available_minutes=15,
    )

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

# ==========================================

