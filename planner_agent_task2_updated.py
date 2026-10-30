# =============================================================================
# planner_agent.py  (Week 2)
# الدور: Action Planner Agent - وكيل تخطيط الواجبات السلوكية (LangGraph + Gemini)
#
# المطلوب في الأسبوع التاني:
#   ربط دالة التخطيط بـ Gemini API فعليا، وتأمين المخرجات بصيغة منظمة (Structured
#   Output)، بحيث الوكيل ياخد (اسم الواجب + الوقت المتاح) ويطلّع جدول لـ 7 أيام،
#   لكل يوم تاريخ ووقت محددين.
#
# اللي اتضاف عن الأسبوع الأول:
#   1) Validator: خطوة تتأكد بالكود إن الخطة سليمة (التواريخ، الأوقات، وإن مجموع
#      الدقائق في اليوم ما يعدّيش الوقت المتاح) بدل ما نثق في الموديل بس.
#   2) Retry: لو الخطة فيها مشاكل، الـ Graph بيرجع للموديل ويديله الأخطاء يصلّحها
#      (لحد 3 محاولات).
#   3) plan_to_calendar_events: دالة بتحوّل الخطة لشكل جاهز لدالة التقويم
#      add_event_to_calendar بتاعة بسمة.
#   4) quran_context: متغير اختياري بيستقبل سياق الآية والتفسير من الـ RAG
#      (rag_retriever.py بتاع مروة) ويتحط في الـ Prompt عشان الواجبات تتبني
#      على معنى الآية. لو مش متبعت، الـ Planner بيشتغل زي الأول.
#
# شكل الـ Graph:
#   START --> planner --> validator --(فيه أخطاء ولسه فيه محاولات)--> planner
#                                    \--(سليمة أو خلصت المحاولات)--> END
# =============================================================================

import os
import re
from datetime import date, timedelta
from typing import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

# -----------------------------------------------------------------------------
# 1) قراءة مفتاح الـ API بشكل آمن
# -----------------------------------------------------------------------------
# المفتاح بييجي من ملف .env (مش مكتوب في الكود) عشان ما يتسرّبش لو الملف اترفع.
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY is missing. Add it to your .env file.")

# -----------------------------------------------------------------------------
# 2) إعدادات ثابتة
# -----------------------------------------------------------------------------
# اسم الموديل. لو ظهر error 404 جربي تحطي في .env سطر: GEMINI_MODEL=gemini-2.5-flash
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

# أقصى عدد محاولات للموديل لو الخطة طلعت فيها أخطاء.
MAX_ATTEMPTS = 3

# عدد أيام الخطة.
PLAN_DAYS = 7


# -----------------------------------------------------------------------------
# 3) شكل الخرج (Structured Output)
# -----------------------------------------------------------------------------
# الموديل بيتجبر يرجّع بالشكل ده بالظبط، فنتجنب النص الحر ونقدر نقرا النتيجة
# بالكود (التقويم، الـ Replanning Agent).

class DayTask(BaseModel):
    """مهمة واحدة في يوم واحد."""
    date: str = Field(description="YYYY-MM-DD")
    time: str = Field(description="HH:MM in 24-hour format")
    duration_min: int = Field(description="Duration in minutes, greater than 0")
    task: str = Field(description="The task text in simple Arabic")


class WeeklyPlan(BaseModel):
    """الخطة الأسبوعية كاملة."""
    summary: str = Field(description="One short Arabic sentence summarizing the plan")
    tasks: list[DayTask]


# -----------------------------------------------------------------------------
# 4) الـ State (البيانات اللي بتتنقل جوه الـ Graph)
# -----------------------------------------------------------------------------
# الحقول الأربعة الأولى هي المدخلات (اتفقنا عليها مع صفاء في state.py).
# باقي الحقول داخلية للـ Planner: بتخلّي الـ Graph يعيد المحاولة لو فيه أخطاء.

class PlannerState(TypedDict):
    quran_context: str       # سياق الآية والتفسير المجلوب من الـ RAG (اختياري)
    action_text: str         # الواجب العملي المختار
    available_minutes: int   # الوقت المتاح كل يوم بالدقائق
    start_date: str          # تاريخ البداية YYYY-MM-DD
    preferred_time: str      # "18:00" أو "morning" أو "after Fajr" أو "any"
    plan: dict               # الخطة الناتجة
    feedback: str            # ملاحظات الـ Validator للموديل في المحاولة اللي بعدها
    attempts: int            # عدد المحاولات اللي حصلت
    warnings: list[str]      # الأخطاء المتبقية (فاضية لو الخطة سليمة)


# -----------------------------------------------------------------------------
# 5) الموديل (Gemini) والـ Prompt
# -----------------------------------------------------------------------------
# temperature=0.3: ردود ثابتة. with_structured_output: بيرجّع كائن WeeklyPlan.
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.3)
structured_llm = llm.with_structured_output(WeeklyPlan)

PROMPT = """You are an Action Planner for a Quran reflection app.
Turn the practical action below into a {plan_days}-day schedule based on the Quranic context provided.

Quranic & Tafsir Context:
{quran_context}

Action: {action_text}
Available time per day: {available_minutes} minutes
Start date: {start_date}
Preferred time: {preferred_time}

Rules:
- The plan must cover exactly {plan_days} consecutive days, starting on the start date.
- The total duration of all tasks in one day must NEVER exceed the available minutes.
- Every task must have a specific date (YYYY-MM-DD), a time (HH:MM, 24-hour), and duration_min.
- Schedule every task at the preferred time. If it is a time of day (e.g. morning, after Fajr),
  convert it to a reasonable HH:MM. If it is "any", choose a suitable time.
- Start small and build gradually; one short task per day is fine.
- Write the task text in simple, warm Arabic.
- Do NOT add any tafsir or religious rulings; only practical steps.
"""
# آخر قاعدة مهمة: المشروع بيفصل بين التفسير المنقول (من مصادر معتمدة عن طريق
# الـ RAG) وبين الاقتراحات العملية، فالـ Planner ممنوع يضيف تفسير من عنده.


# -----------------------------------------------------------------------------
# 6) التحقق من الخطة (Validation) - بالكود مش بالموديل
# -----------------------------------------------------------------------------
TIME_PATTERN = re.compile(r"([01]\d|2[0-3]):[0-5]\d")  # HH:MM


def validate_plan(plan: dict, available_minutes: int, start_date: str) -> list[str]:
    """بترجّع قايمة أخطاء. لو القايمة فاضية يبقى الخطة سليمة."""
    errors: list[str] = []
    tasks = plan.get("tasks", [])

    if not tasks:
        return ["The plan has no tasks."]

    start = date.fromisoformat(start_date)
    end = start + timedelta(days=PLAN_DAYS - 1)
    minutes_per_day: dict[str, int] = {}

    for t in tasks:
        # 1) التاريخ لازم يكون بصيغة صحيحة وداخل نطاق الـ 7 أيام.
        try:
            d = date.fromisoformat(t["date"])
        except ValueError:
            errors.append(f"Invalid date format: {t['date']} (use YYYY-MM-DD).")
            continue
        if not (start <= d <= end):
            errors.append(f"Date {t['date']} is outside {start} to {end}.")

        # 2) الوقت لازم يكون HH:MM.
        if not TIME_PATTERN.fullmatch(t["time"]):
            errors.append(f"Invalid time: {t['time']} (use HH:MM, 24-hour).")

        # 3) المدة لازم تكون أكبر من صفر.
        if t["duration_min"] <= 0:
            errors.append(f"Duration must be > 0 on {t['date']}.")

        # 4) نجمع دقايق كل يوم.
        minutes_per_day[t["date"]] = minutes_per_day.get(t["date"], 0) + t["duration_min"]

    # 5) مجموع دقايق اليوم ما يعدّيش الوقت المتاح.
    for day, total in minutes_per_day.items():
        if total > available_minutes:
            errors.append(
                f"On {day} total is {total} min, but only {available_minutes} min are available."
            )

    # 6) لازم كل يوم من الـ 7 أيام يبقى فيه مهمة على الأقل.
    covered = set(minutes_per_day)
    for i in range(PLAN_DAYS):
        day = (start + timedelta(days=i)).isoformat()
        if day not in covered:
            errors.append(f"No task scheduled on {day}.")

    return errors


# -----------------------------------------------------------------------------
# 7) الـ Nodes (خطوات الـ Graph)
# -----------------------------------------------------------------------------
def planner_node(state: PlannerState) -> dict:
    """بتبعت الـ Prompt لـ Gemini وترجّع الخطة."""
    # لو مفيش سياق من الـ RAG بنحط جملة بديلة عشان الـ Prompt ما يبقاش فيه فراغ.
    values = {
        **state,
        "quran_context": state["quran_context"] or "(No Quranic context provided.)",
    }
    prompt = PROMPT.format(plan_days=PLAN_DAYS, **values)

    # لو دي محاولة تانية، بنضيف الأخطاء السابقة عشان الموديل يصلّحها.
    if state["feedback"]:
        prompt += (
            "\nYour previous plan had these problems. Fix all of them:\n"
            + state["feedback"]
        )

    result = structured_llm.invoke(prompt)
    return {"plan": result.model_dump(), "attempts": state["attempts"] + 1}


def validator_node(state: PlannerState) -> dict:
    """بتفحص الخطة بالكود وتسجّل الأخطاء في الـ State."""
    errors = validate_plan(state["plan"], state["available_minutes"], state["start_date"])
    feedback = "\n".join(f"- {e}" for e in errors)
    return {"warnings": errors, "feedback": feedback}


def route_after_validation(state: PlannerState) -> str:
    """بتقرر: نعيد المحاولة ولا ننتهي؟"""
    if state["warnings"] and state["attempts"] < MAX_ATTEMPTS:
        return "planner"
    return END


# -----------------------------------------------------------------------------
# 8) بناء الـ Graph
# -----------------------------------------------------------------------------
graph = StateGraph(PlannerState)
graph.add_node("planner", planner_node)
graph.add_node("validator", validator_node)
graph.add_edge(START, "planner")
graph.add_edge("planner", "validator")
# الـ Edge الشرطي: بعد الفحص يا نرجع للـ planner يا ننتهي.
graph.add_conditional_edges("validator", route_after_validation, ["planner", END])
app = graph.compile()


# -----------------------------------------------------------------------------
# 9) الدالة الرئيسية اللي باقي التيم بيستدعيها
# -----------------------------------------------------------------------------
def plan_action(
    action_text: str,
    available_minutes: int,
    quran_context: str = "",
    start_date: str | None = None,
    preferred_time: str = "any",
) -> dict:
    """
    بتاخد الواجب والوقت المتاح (وسياق الآية من الـ RAG لو متاح) وبترجّع dict فيه:
      summary  : ملخص الخطة
      tasks    : قايمة مهام (date, time, duration_min, task)
      warnings : أخطاء متبقية بعد المحاولات (قايمة فاضية = الخطة سليمة)
    """
    start_date = start_date or date.today().isoformat()

    out = app.invoke({
        "quran_context": quran_context,
        "action_text": action_text,
        "available_minutes": available_minutes,
        "start_date": start_date,
        "preferred_time": preferred_time,
        "plan": {},
        "feedback": "",
        "attempts": 0,
        "warnings": [],
    })

    return {**out["plan"], "warnings": out["warnings"]}


# -----------------------------------------------------------------------------
# 10) التكامل مع التقويم (Google Calendar)
# -----------------------------------------------------------------------------
def plan_to_calendar_events(plan: dict) -> list[dict]:
    """
    بتحوّل مهام الخطة لقايمة أحداث بنفس أسماء مدخلات دالة بسمة:
      add_event_to_calendar(summary, start_time, duration_minutes)
    start_time بصيغة ISO بدون timezone، مثلا 2026-10-25T18:00:00
    (لو بسمة محتاجة صيغة تانية نعدّل السطر ده بس).
    """
    return [
        {
            "summary": t["task"],
            "start_time": f"{t['date']}T{t['time']}:00",
            "duration_minutes": t["duration_min"],
        }
        for t in plan["tasks"]
    ]


# -----------------------------------------------------------------------------
# 11) تجربة (بتشتغل بس لما نشغّل الملف مباشرة)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    # حالات تجربة مختلفة: واجب + وقت متاح + وقت مفضل.
    test_cases = [
        ("اذكر الله 5 دقائق بعد كل صلاة", 15, "18:00"),
        ("صلة الرحم: اتصل بقريب كل يوم", 10, "morning"),
        ("قراءة صفحة من التفسير وتدبّرها", 30, "after Fajr"),
    ]

    for action, minutes, pref in test_cases:
        print("=" * 60)
        print(f"Action: {action} | {minutes} min/day | preferred: {pref}")
        result = plan_action(action, minutes, preferred_time=pref)
        print(json.dumps(result, ensure_ascii=False, indent=2))

        # عرض شكل الأحداث الجاهزة للتقويم (أول حدث بس للاختصار).
        events = plan_to_calendar_events(result)
        print("First calendar event:", events[0] if events else "none")
