import json
from pathlib import Path
from typing import Dict, Any, Optional

# تحديد مسار ملف بيانات سورة البقرة (الجزء الأول)
BASE_DIR = Path(__file__).parent
QURAN_DATA_PATH = BASE_DIR / "tafsir_baqarah_part1.json"


def load_quran_knowledge_base() -> list[Dict[str, Any]]:
    """تحميل قاعدة بيانات القرآن والتفسير الميسر والواجبات السلوكية."""
    if not QURAN_DATA_PATH.exists():
        print(f"⚠️ تحذير: ملف البيانات غير موجود في المسار: {QURAN_DATA_PATH}")
        return []

    try:
        with open(QURAN_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return data.get("verses", data.get("ayahs", []))
            return []
    except Exception as e:
        print(f"❌ خطأ أثناء قراءة ملف البيانات: {e}")
        return []


def get_verse_context(verse_number: int) -> Optional[Dict[str, Any]]:
    """استرجاع بيانات آية محددة برقمها من 1 إلى 141."""
    if not (1 <= verse_number <= 141):
        print("⚠️ رقم الآية خارج نطاق الجزء الأول (1-141).")
        return None

    knowledge_base = load_quran_knowledge_base()
    for item in knowledge_base:
        if item.get("verse_number") == verse_number:
            return item
            
    return None


def format_rag_prompt_context(verse_number: int) -> str:
    """
    تجهيز وتنسيق السياق القرآني (الآية والتفسير الموثوق فقط)
    لإرساله إلى generate_reflection و planner_agent.
    """
    verse_data = get_verse_context(verse_number)
    
    if not verse_data:
        return ""

    arabic_text = verse_data.get("arabic_text", "").strip()
    tafsir_text = verse_data.get("tafsir_text", "").strip()

    if not arabic_text or not tafsir_text:
        return ""

    formatted_context = (
        f"السورة: البقرة - الآية رقم: {verse_number}\n"
        f"نص الآية: ﴿{arabic_text}﴾\n"
        f"التفسير الميسر: {tafsir_text}"
    )
    return formatted_context.strip()


# ==========================================
# اختبار الدالة (Smoke Test)
# ==========================================
if __name__ == "__main__":
    print("🚀 جاري اختبار مسترجع البيانات RAG Retriever...\n")
    
    # تجربة استرجاع بيانات الآية رقم 3
    test_verse = 3
    context_output = format_rag_prompt_context(test_verse)
    
    print(context_output)