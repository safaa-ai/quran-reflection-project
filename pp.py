
import os
import streamlit as st

st.set_page_config(
    page_title="Quran Reflection & Action System",
    page_icon="📖",
    layout="centered"
)

# Load Gemini API Key before importing the agents
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("GOOGLE_API_KEY is missing from Streamlit Secrets.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

from agent_graph import app as workflow

st.title("Quran Reflection & Action System")
st.write("Welcome to our Quran Reflection project!")

st.header("Create Your Weekly Plan")

verse_number = st.number_input(
    "Choose an Ayah from Surah Al-Baqarah (1–141)",
    min_value=1,
    max_value=141,
    value=1,
    step=1
)

selected_action = st.text_input(
    "What action do you want to practice?"
)

available_minutes = st.number_input(
    "Available minutes per day",
    min_value=5,
    max_value=180,
    value=30,
    step=5
)

start_date = st.date_input("Start Date")

preferred_time = st.selectbox(
    "Preferred Time",
    ["any", "morning", "afternoon", "evening"]
)

if st.button("Generate My Plan"):

    if not selected_action.strip():
        st.warning("Please enter an action first.")

    else:
        initial_state = {
            "verse_number": int(verse_number),
            "selected_action": selected_action.strip(),
            "available_minutes": int(available_minutes),
            "start_date": str(start_date),
            "duration_days": 7,
            "preferred_time": preferred_time,
            "overdue_tasks": []
        }

        try:
            with st.spinner("Generating your plan..."):
                result = workflow.invoke(initial_state)

            if result.get("error"):
                st.error(result["error"])

            else:
                st.success("Your plan was generated!")

                if result.get("quran_context"):
                    st.subheader("Quran Context")
                    st.write(result["quran_context"])

                if result.get("plan_summary"):
                    st.subheader("Plan Summary")
                    st.write(result["plan_summary"])

                st.subheader("Weekly Plan")
                weekly_plan = result.get("weekly_plan", [])

                if weekly_plan:
                    st.json(weekly_plan)
                else:
                    st.info("No weekly tasks were returned.")

                if result.get("plan_warnings"):
                    st.subheader("Plan Warnings")
                    for warning in result["plan_warnings"]:
                        st.warning(warning)

                if result.get("revised_plan"):
                    st.subheader("Revised Plan")
                    st.json(result["revised_plan"])

                if result.get("replanning_message"):
                    st.info(result["replanning_message"])

        except Exception as e:
            st.error(f"An error occurred: {e}")
