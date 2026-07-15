import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLP data dictionaries 
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize the NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # 1. Convert to lowercase
    text = text.lower()
    # 2. Remove URLs (common in news articles)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # 3. Remove punctuation and numbers (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 4. Tokenize (split into a list of words)
    tokens = word_tokenize(text)
    # 5. Remove stop words and reduce words to their base form
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # 6. Rejoin into a single string
    return " ".join(cleaned)

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
        # Run the cleaning pipeline
        cleaned_result = clean_text(user_input)
        
        # Display the results for testing
        st.success("Text successfully processed through the NLP pipeline!")
        st.write("**Before (Raw):**", user_input[:300] + "...")
        st.write("**After (Cleaned):**", cleaned_result[:300] + "...")
    else:
        st.warning("Please enter a longer article for accurate analysis.")