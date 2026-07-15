import streamlit as st

# 1. Page Configuration
st.set_page_config(page_title="Fake News Detector", layout="centered")

# 2. UI Header
st.title("📰 Fake News Detection Engine")
st.write("Paste a news article below to analyze its linguistic patterns and cross-reference fact-checks.")

# 3. User Input Box
user_input = st.text_area("Article Text", height=300, placeholder="Paste the article here...")

# 4. Analyze Button
if st.button("Analyze Article"):
    if len(user_input) > 50:
        st.info("The NLP engine and API connection will run here.")
    else:
        st.warning("Please enter a longer article for accurate analysis.")