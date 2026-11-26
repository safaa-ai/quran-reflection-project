# =============================================================================
# planner_agent.py  (Week 3)
# الدور: Action Planner Agent - وكيل تخطيط الواجبات السلوكية (LangGraph + Gemini)
#
# المطلوب في الأسبوع التالت:
#   1) تحديث الـ Prompt: الموديل يكتب فقرة تدبرية دافئة بتجمع مقاصد الآية وتنزّل
#      معانيها على واقع المسلم المعاصر، ويستخرج 3 إلى 4 اقتراحات سلوكية عملية.
#   2) تعديل الـ Structured Output: كائن فيه reflection_text و suggested_actions.
#   3) دعم الجدول المخصص: استقبال العمل المختار (مقترح أو مكتوب بإيد المستخدم)
#      مع تاريخ البداية والنهاية (أو المدة) وبناء التوزيع الزمني على المدة دي.
#
# اللي اتضاف عن الأسبوع التاني:
#   - generate_reflection(quran_context): الخطوة الأولى (بعد الـ RAG) وبترجّع
#       reflection_text + suggested_actions للواجهة.
#   - plan_action(...): بقت بتقبل end_date أو duration_days و is_custom، وعدد
#       الأيام بقى ديناميكي (مش 7 ثابتة). لو ما اتبعتش end_date ولا duration_days
#       بتشتغل 7 أيام زي الأول (متوافقة مع الكود القديم).
#   - Validator بقى بيحسب على عدد الأيام الفعلي بدل PLAN_DAYS الثابتة.
#
# الفلو الكامل:
#   RAG --> generate_reflection --> (الواجهة: المستخدم يختار مقترح أو يكتب عمله)
#       --> plan_action(start_date, end_date) --> plan_to_calendar_events
#
# شكل الـ Graph (الجدولة):
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

# أقصى عدد محاولات للموديل لو الناتج طلع فيه أخطاء.
MAX_ATTEMPTS = 3

# المدة الافتراضية للخطة (أسبوع) لو المستخدم ما حددش تاريخ نهاية ولا مدة.
DEFAULT_PLAN_DAYS = 7

# أقصى مدة مسموحة للخطة (شهر) عشان الـ Prompt والخرج ما يكبروش زيادة.
MAX_PLAN_DAYS = 31

# حدود عدد الاقتراحات السلوكية في خطوة التدبر.
MIN_SUGGESTIONS = 3
MAX_SUGGESTIONS = 4


# -----------------------------------------------------------------------------
# 3) شكل الخرج (Structured Output)
# -----------------------------------------------------------------------------
# الموديل بيتجبر يرجّع بالشكل ده بالظبط، فنتجنب النص الحر ونقدر نقرا النتيجة
# بالكود (الواجهة، التقويم، الـ Replanning Agent).

class ReflectionOutput(BaseModel):
    """خرج خطوة التدبر: فقرة تدبرية + اقتراحات سلوكية."""
    reflection_text: str = Field(
        description=(
            "One warm Arabic paragraph (4-6 sentences) that explains the objectives "
            "(maqasid) of the verses and applies their meanings to the daily reality "
            "of a Muslim living today."
        )
    )
    suggested_actions: list[str] = Field(
        description=(
            "3 to 4 practical, specific behavioral actions in simple Arabic, "
            "one short sentence each."
        )
    )


class DayTask(BaseModel):
    """مهمة واحدة في يوم واحد."""
    date: str = Field(description="YYYY-MM-DD")
    time: str = Field(description="HH:MM in 24-hour format")
    duration_min: int = Field(description="Duration in minutes, greater than 0")
    task: str = Field(description="The task text in simple Arabic")


class WeeklyPlan(BaseModel):
    """الخطة كاملة (أسبوع أو أكتر حسب تاريخ البداية والنهاية)."""
    summary: str = Field(description="One short Arabic sentence summarizing the plan")
    tasks: list[DayTask]


# -----------------------------------------------------------------------------
# 4) الـ State (البيانات اللي بتتنقل جوه الـ Graph بتاع الجدولة)
# -----------------------------------------------------------------------------
# الحقول الأولى هي المدخلات (اتفقنا عليها مع صفاء في state.py).
# باقي الحقول داخلية للـ Planner: بتخلّي الـ Graph يعيد المحاولة لو فيه أخطاء.

class PlannerState(TypedDict):
    quran_context: str       # سياق الآية والتفسير المجلوب من الـ RAG (اختياري)
    action_text: str         # الواجب العملي المختار (مقترح أو مخصص)
    is_custom: bool          # True لو المستخدم كتب العمل بنفسه
    available_minutes: int   # الوقت المتاح كل يوم بالدقائق
    start_date: str          # تاريخ البداية YYYY-MM-DD
    end_date: str            # تاريخ النهاية YYYY-MM-DD (شامل)
    plan_days: int           # عدد أيام الخطة (محسوب من البداية والنهاية)
    preferred_time: str      # "18:00" أو "morning" أو "after Fajr" أو "any"
    plan: dict               # الخطة الناتجة
    feedback: str            # ملاحظات الـ Validator للموديل في المحاولة اللي بعدها
    attempts: int            # عدد المحاولات اللي حصلت
    warnings: list[str]      # الأخطاء المتبقية (فاضية لو الخطة سليمة)


# -----------------------------------------------------------------------------
# 5) الموديلات (Gemini) والـ Prompts
# -----------------------------------------------------------------------------
# موديل الجدولة: temperature=0.3 عشان الردود تبقى ثابتة.
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.3)
structured_llm = llm.with_structured_output(WeeklyPlan)

# موديل التدبر: temperature أعلى شوية (0.6) عشان الفقرة تبقى دافية وطبيعية.
reflection_llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.6)
structured_reflection_llm = reflection_llm.with_structured_output(ReflectionOutput)

# ---- Prompt التدبر (الخطوة الأولى بعد الـ RAG) ----
REFLECTION_PROMPT = """You are a warm reflection guide for a Quran reflection app.
Use ONLY the Quranic and Tafsir context below.

Quranic & Tafsir Context:
{quran_context}

Produce two things:

1) reflection_text: ONE flowing Arabic paragraph (4 to 6 sentences) that
   - explains the objectives (maqasid) and core meanings of these verses, then
   - applies those meanings to the daily reality of a Muslim living today
     (for example: work pressure, family, phone and social media, stress, relationships).
   Make it feel personal and encouraging, not like a lecture.

2) suggested_actions: {min_s} to {max_s} practical behavioral actions that flow from the paragraph.

Rules:
- Base the paragraph only on the context above. Do NOT invent hadith, narrations,
  scholar names, or tafsir that is not in the context.
- Do NOT give religious rulings (fatwas). Keep the actions practical, not legal.
- Write in simple, warm Arabic.
- Each action must be: one short sentence, starting with a verb, concrete and doable in
  5 to 30 minutes a day, and clearly different from the other actions.
"""

# ---- Prompt الجدولة (بعد ما المستخدم يختار العمل ويحدد التواريخ) ----
PROMPT = """You are an Action Planner for a Quran reflection app.
Turn the practical action below into a {plan_days}-day schedule based on the Quranic context provided.

Quranic & Tafsir Context:
{quran_context}

Action: {action_text}
Action source: {action_source}
Available time per day: {available_minutes} minutes
Start date: {start_date}
End date: {end_date}
Preferred time: {preferred_time}

Rules:
- The plan must cover exactly {plan_days} consecutive days, from the start date to the end date (both included).
- The total duration of all tasks in one day must NEVER exceed the available minutes.
- Every task must have a specific date (YYYY-MM-DD), a time (HH:MM, 24-hour), and duration_min.
- Schedule every task at the preferred time. If it is a time of day (e.g. morning, after Fajr),
  convert it to a reasonable HH:MM. If it is "any", choose a suitable time.
- Start small and build gradually; one short task per day is fine. For longer plans, vary the
  wording of the daily task a little so it does not feel repetitive.
- Write the task text in simple, warm Arabic.
- Do NOT add any tafsir or religious rulings; only practical steps.
"""
# آخر قاعدة مهمة: المشروع بيفصل بين التفسير المنقول (من مصادر معتمدة عن طريق
# الـ RAG) وبين الاقتراحات العملية، فالـ Planner ممنوع يضيف تفسير من عنده.


# -----------------------------------------------------------------------------
# 6) التحقق (Validation) - بالكود مش بالموديل
# -----------------------------------------------------------------------------
TIME_PATTERN = re.compile(r"([01]\d|2[0-3]):[0-5]\d")  # HH:MM


def resolve_dates(
    start_date: str | None,
    end_date: str | None = None,
    duration_days: int | None = None,
) -> tuple[str, str, int]:
    """
    بتحدد تاريخ البداية والنهاية وعدد الأيام. الأولوية:
      1) end_date لو اتبعت  2) duration_days (7 = أسبوع، 30 = شهر)  3) 7 أيام.
    بترجّع (start_date, end_date, plan_days) وبتعمل ValueError لو المدخلات غلط.
    """
    start = date.fromisoformat(start_date) if start_date else date.today()

    if end_date:
        end = date.fromisoformat(end_date)
    else:
        end = start + timedelta(days=(duration_days or DEFAULT_PLAN_DAYS) - 1)

    plan_days = (end - start).days + 1
    if plan_days < 1:
        raise ValueError("end_date must be on or after start_date.")
    if plan_days > MAX_PLAN_DAYS:
        raise ValueError(f"The plan cannot be longer than {MAX_PLAN_DAYS} days.")

    return start.isoformat(), end.isoformat(), plan_days


def validate_plan(
    plan: dict,
    available_minutes: int,
    start_date: str,
    plan_days: int = DEFAULT_PLAN_DAYS,
) -> list[str]:
    """بترجّع قايمة أخطاء. لو القايمة فاضية يبقى الخطة سليمة."""
    errors: list[str] = []
    tasks = plan.get("tasks", [])

    if not tasks:
        return ["The plan has no tasks."]

    start = date.fromisoformat(start_date)
    end = start + timedelta(days=plan_days - 1)
    minutes_per_day: dict[str, int] = {}

    for t in tasks:
        # 1) التاريخ لازم يكون بصيغة صحيحة وداخل نطاق الخطة.
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

    # 6) لازم كل يوم من أيام الخطة يبقى فيه مهمة على الأقل.
    covered = set(minutes_per_day)
    for i in range(plan_days):
        day = (start + timedelta(days=i)).isoformat()
        if day not in covered:
            errors.append(f"No task scheduled on {day}.")

    return errors


def validate_reflection(data: dict) -> list[str]:
    """بتتأكد إن التدبر والاقتراحات بالشكل المطلوب. قايمة فاضية = سليم."""
    errors: list[str] = []

    text = (data.get("reflection_text") or "").strip()
    if len(text) < 80:
        errors.append("reflection_text is too short; write a full paragraph of 4-6 sentences.")
    elif len(text) > 1500:
        errors.append("reflection_text is too long; keep it to one paragraph of 4-6 sentences.")

    actions = [a.strip() for a in data.get("suggested_actions", []) if a and a.strip()]
    if not (MIN_SUGGESTIONS <= len(actions) <= MAX_SUGGESTIONS):
        errors.append(
            f"Provide {MIN_SUGGESTIONS} to {MAX_SUGGESTIONS} suggested actions (got {len(actions)})."
        )
    if len(set(actions)) != len(actions):
        errors.append("Suggested actions must be different from each other.")
    if any(len(a) > 160 for a in actions):
        errors.append("Each action must be one short sentence.")

    return errors


# -----------------------------------------------------------------------------
# 7) الـ Nodes (خطوات الـ Graph بتاع الجدولة)
# -----------------------------------------------------------------------------
def planner_node(state: PlannerState) -> dict:
    """بتبعت الـ Prompt لـ Gemini وترجّع الخطة."""
    # لو مفيش سياق من الـ RAG بنحط جملة بديلة عشان الـ Prompt ما يبقاش فيه فراغ.
    values = {
        **state,
        "quran_context": state["quran_context"] or "(No Quranic context provided.)",
        "action_source": (
            "written by the user; keep its intent exactly"
            if state["is_custom"]
            else "suggested by the app"
        ),
    }
    prompt = PROMPT.format(**values)

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
    errors = validate_plan(
        state["plan"], state["available_minutes"], state["start_date"], state["plan_days"]
    )
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
# 9) الدوال الرئيسية اللي باقي التيم بيستدعيها
# -----------------------------------------------------------------------------
def generate_reflection(quran_context: str) -> dict:
    """
    الخطوة الأولى (بعد الـ RAG): بتاخد نص الآية والتفسير وبترجّع dict فيه:
      reflection_text   : فقرة تدبرية مربوطة بواقع المسلم اليوم
      suggested_actions : قايمة فيها 3 إلى 4 اقتراحات سلوكية
      warnings          : أخطاء متبقية بعد المحاولات (قايمة فاضية = سليم)
    """
    if not quran_context or not quran_context.strip():
        raise ValueError("quran_context is required to generate a reflection.")

    base_prompt = REFLECTION_PROMPT.format(
        quran_context=quran_context, min_s=MIN_SUGGESTIONS, max_s=MAX_SUGGESTIONS
    )
    feedback = ""
    data: dict = {}
    errors: list[str] = []

    for _ in range(MAX_ATTEMPTS):
        prompt = base_prompt
        if feedback:
            prompt += "\nYour previous answer had these problems. Fix all of them:\n" + feedback

        data = structured_reflection_llm.invoke(prompt).model_dump()
        errors = validate_reflection(data)
        if not errors:
            break
        feedback = "\n".join(f"- {e}" for e in errors)

    return {**data, "warnings": errors}


def plan_action(
    action_text: str,
    available_minutes: int,
    quran_context: str = "",
    start_date: str | None = None,
    preferred_time: str = "any",
    end_date: str | None = None,
    duration_days: int | None = None,
    is_custom: bool = False,
) -> dict:
    """
    بتاخد العمل المختار (مقترح أو مخصص) والوقت المتاح والتواريخ، وبترجّع dict فيه:
      summary    : ملخص الخطة
      tasks      : قايمة مهام (date, time, duration_min, task)
      start_date : تاريخ البداية
      end_date   : تاريخ النهاية (عشان يتخزن في الداتابيز عند مي)
      warnings   : أخطاء متبقية بعد المحاولات (قايمة فاضية = الخطة سليمة)

    المدة: end_date لو اتبعت، وإلا duration_days (7 = أسبوع، 30 = شهر)، وإلا 7 أيام.
    """
    start_date, end_date, plan_days = resolve_dates(start_date, end_date, duration_days)

    out = app.invoke({
        "quran_context": quran_context,
        "action_text": action_text,
        "is_custom": is_custom,
        "available_minutes": available_minutes,
        "start_date": start_date,
        "end_date": end_date,
        "plan_days": plan_days,
        "preferred_time": preferred_time,
        "plan": {},
        "feedback": "",
        "attempts": 0,
        "warnings": [],
    })

    return {
        **out["plan"],
        "start_date": start_date,
        "end_date": end_date,
        "warnings": out["warnings"],
    }


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

    # سياق تجريبي (في المشروع الحقيقي بييجي من rag_retriever.py بتاع مروة).
    sample_context = (
        "الآيات: ﴿وَقُولُوا لِلنَّاسِ حُسْنًا﴾ (البقرة: 83)\n"
        "التفسير: أمرٌ بحسن الكلام مع الناس جميعًا، ولين الخطاب، وطيب المعاملة."
    )

    # 1) خطوة التدبر: فقرة + اقتراحات.
    print("=" * 60)
    reflection = generate_reflection(sample_context)
    print(json.dumps(reflection, ensure_ascii=False, indent=2))

    # 2) عمل مقترح لمدة أسبوع (تاريخ النهاية بيتحسب تلقائي).
    print("=" * 60)
    chosen = reflection["suggested_actions"][0]
    week_plan = plan_action(
        chosen, 15, quran_context=sample_context, preferred_time="18:00", is_custom=False
    )
    print(json.dumps(week_plan, ensure_ascii=False, indent=2))

    # 3) عمل مخصص بتاريخ بداية ونهاية (حوالي أسبوعين).
    print("=" * 60)
    start = date.today()
    custom_plan = plan_action(
        "أتصل بجدتي كل يوم وأسأل عن أحوالها",
        10,
        quran_context=sample_context,
        start_date=start.isoformat(),
        end_date=(start + timedelta(days=13)).isoformat(),
        preferred_time="after Fajr",
        is_custom=True,
    )
    print(json.dumps(custom_plan, ensure_ascii=False, indent=2))

    # عرض شكل الأحداث الجاهزة للتقويم (أول حدث بس للاختصار).
    events = plan_to_calendar_events(custom_plan)
    print("First calendar event:", events[0] if events else "none")
