
import streamlit as st

st.set_page_config(
    page_title="Quran Reflection & Action System",
    page_icon="📖",
    layout="centered"
)

st.title("Quran Reflection & Action System")

st.write("Welcome to our Quran Reflection project!")

verse_number = st.number_input(
    "Choose a verse number from Surah Al-Baqarah",
    min_value=1,
    max_value=141,
    value=1
)

st.write(f"You selected verse number: {verse_number}")

st.success("The application is running successfully!")