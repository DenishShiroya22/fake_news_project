import streamlit as st
import joblib
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Web App Configuration ---
st.set_page_config(page_title="Hybrid Misinformation Engine", layout="wide")
st.title("🛡️ Hybrid Misinformation Detection Engine")

# --- Load Engine 1 (Linguistic AI) ---
@st.cache_resource
def load_engine_1():
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

# --- Load Engine 2 (Vector Knowledge Base) ---
@st.cache_resource
def load_engine_2():
    with open('trusted_knowledge.json', 'r') as f:
        kb = json.load(f)
    
    facts = [item['fact'] for item in kb]
    
    # We train a NEW vector space purely for the local facts
    kb_vectorizer = TfidfVectorizer(stop_words='english')
    kb_matrix = kb_vectorizer.fit_transform(facts)
    return kb, kb_vectorizer, kb_matrix

# Initialize Both Engines safely
try:
    lr_model, lr_vectorizer = load_engine_1()
    knowledge_base, kb_vectorizer, kb_matrix = load_engine_2()
except FileNotFoundError:
    st.error("Missing critical files. Ensure model.pkl, vectorizer.pkl, and trusted_knowledge.json are in the folder.")
    st.stop()

# --- User Interface ---
st.write("Paste a news article or claim below to run it through our dual-verification architecture.")
user_input = st.text_area("News Text:", height=150)

if st.button("Run Hybrid Analysis"):
    if user_input.strip():
        
        # Split the screen into two columns
        col1, col2 = st.columns(2)
        
        # ==========================================
        # ENGINE 1: LINGUISTIC ANALYSIS
        # ==========================================
        with col1:
            st.subheader("🧠 Engine 1: Linguistic Analysis")
            # Map the user's text into the 61,000-row dictionary
            user_vec_1 = lr_vectorizer.transform([user_input])
            prediction = lr_model.predict(user_vec_1)[0]
            probability = lr_model.predict_proba(user_vec_1)[0]
            
            if prediction == 1:
                st.success("**Verdict:** Professional / Reliable Structure")
                st.write(f"**Confidence:** {probability[1]*100:.1f}%")
            else:
                st.error("**Verdict:** Sensationalized / Clickbait")
                st.write(f"**Confidence:** {probability[0]*100:.1f}%")

        # ==========================================
        # ENGINE 2: SEMANTIC FACT CHECK
        # ==========================================
        with col2:
            st.subheader("📚 Engine 2: Local Fact Check")
            
            # Map the user's text into the JSON dictionary
            user_vec_2 = kb_vectorizer.transform([user_input])
            
            # The mathematical algorithm you just learned
            similarities = cosine_similarity(user_vec_2, kb_matrix).flatten()
            best_match_idx = np.argmax(similarities)
            best_score = similarities[best_match_idx]
            
            # Threshold check: Is the angle close enough to be a match?
            if best_score > 0.15: 
                matched_fact = knowledge_base[best_match_idx]
                st.info(f"**Verified Document Found** (Match: {best_score*100:.1f}%)")
                st.write(f"**Source:** {matched_fact['source']}")
                st.write(f"**Fact:** {matched_fact['fact']}")
            else:
                st.warning("**No verified documents found.**")
                st.write("This claim does not mathematically match anything in our local database.")
                st.write(f"(Closest similarity was only {best_score*100:.1f}%)")