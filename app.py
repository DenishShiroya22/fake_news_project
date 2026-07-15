import streamlit as st
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# 1. Download Required NLP Data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# 2. UI Configuration
st.set_page_config(page_title="Fake News Detector", layout="centered")

# 3. Cache the Model Loading (so it only loads once)
@st.cache_resource
def load_models():
    try:
        vec = joblib.load('vectorizer.pkl')
        mod = joblib.load('model.pkl')
        return vec, mod
    except FileNotFoundError:
        st.error("Model files not found. Please run train_model.py first.")
        st.stop()

vectorizer, model = load_models()

# 4. Text Cleaning Function (Must match the training logic exactly)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned)

# 5. Web Interface
st.title("📰 Fake News Detection Engine")
st.write("Paste a news article or headline below to analyze its linguistic patterns.")

user_input = st.text_area("Article Text", height=200, placeholder="Paste the article here...")

if st.button("Analyze Article"):
    if len(user_input) > 20:
        # A. Preprocess the raw input
        cleaned_input = clean_text(user_input)
        
        # B. Convert to Math (Note: transform expects a list of strings)
        vectorized_input = vectorizer.transform([cleaned_input])
        
        # C. Make Predictions
        prediction = model.predict(vectorized_input)[0]
        probabilities = model.predict_proba(vectorized_input)[0]
        
        # Determine confidence score based on the predicted class
        if prediction == 1:
            confidence = probabilities[1] * 100
            st.success("### Verdict: Likely Reliable News")
            st.write(f"**Model Confidence:** {confidence:.1f}% sure this is factual reporting.")
        else:
            confidence = probabilities[0] * 100
            st.warning("### Verdict: High Risk of Misinformation / Clickbait")
            st.write(f"**Model Confidence:** {confidence:.1f}% sure this uses manipulative language.")
            
    else:
        st.warning("Please enter a longer piece of text for accurate analysis.")