import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# 1. Safely Download Required NLP Data (Prevents the LookupError)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# 2. Initialize NLP Tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Standardizes text by removing noise and lemmatizing words."""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)           # Keep only letters
    tokens = word_tokenize(text)                      # Split into words
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned)

# 3. Mock Dataset (1 = Reliable, 0 = Fake/Clickbait)
print("Loading data...")
data = {
    'text': [
        "The stock market saw a 5% increase today following the Federal Reserve's announcement.",
        "BREAKING: Alien spaceship lands in Washington! Government hiding the truth!!!",
        "New study shows that eating apples can improve long-term heart health.",
        "You won't believe what this celebrity did! Click here to find out the shocking secret!",
        "The local city council voted 4-3 to approve the new highway funding project.",
        "Miracle cure for all diseases discovered! Doctors hate him! Buy now!"
    ],
    'label': [1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# 4. Data Processing Pipeline
print("Cleaning text data (this may take a moment on large datasets)...")
df['cleaned_text'] = df['text'].apply(clean_text)

print("Converting text to TF-IDF vectors...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

# 5. Model Training
print("Training the Logistic Regression model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Export Pipeline Assets
print("Saving model and vectorizer to disk...")
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(model, 'model.pkl')
print("Done! vectorizer.pkl and model.pkl successfully created.")