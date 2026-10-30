import os
import streamlit as st

st.set_page_config(
    page_title="Quran Reflection & Action System",
    page_icon="📖",
    layout="centered"
)

# Load Gemini API Key before importing the agents
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

from agent_graph import app as workflow

st.title("Quran Reflection & Action System")

st.write("Welcome to our Quran Reflection project!")

st.header("Create Your Weekly Plan")

selected_action = st.text_input(
    "What action do you want to practice?"
)

available_minutes = st.number_input(
    "Available minutes per day",
    min_value=5,
    max_value=180,
    value=30
)

tafsir_text = st.text_area(
    "Quran Tafsir Context"
)

start_date = st.date_input(
    "Start Date"
)

preferred_time = st.selectbox(
    "Preferred Time",
    ["any", "morning", "afternoon", "evening"]
)

if st.button("Generate My Plan"):

    if not selected_action.strip():
        st.warning("Please enter an action first.")

    else:
        initial_state = {
            "selected_action": selected_action,
            "available_minutes": available_minutes,
            "tafsir_text": tafsir_text,
            "start_date": str(start_date),
            "preferred_time": preferred_time,
            "overdue_tasks": []
        }

        try:
            with st.spinner("Generating your plan..."):

                result = workflow.invoke(initial_state)

            st.success("Your plan was generated successfully!")

            st.subheader("Weekly Plan")

            st.write(result.get("weekly_plan", []))

            if result.get("revised_plan"):
                st.subheader("Revised Plan")
                st.write(result["revised_plan"])

        except Exception as e:
            st.error(f"An error occurred: {e}")
