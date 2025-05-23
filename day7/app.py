# File: app.py

import streamlit as st # Keep this at the top
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# --- Streamlit UI config MUST be the first Streamlit command ---
st.set_page_config(page_title="Movie Review Sentiment Analyzer", layout="centered")

# --- NLTK Downloads (Ensures stopwords are available when the app runs) ---
# Moved after set_page_config
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    st.info("Downloaded NLTK stopwords. This happens once on first run.")

# --- Define the Preprocessing Function ---
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    processed_words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(processed_words)

# --- Load the Model and Vectorizer ---
model_dir = 'models' # Ensure this path is correct for your environment
model_path = os.path.join(model_dir, 'best_logistic_regression_model.joblib')
vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.joblib')

@st.cache_resource
def load_resources():
    try:
        loaded_model = joblib.load(model_path)
        loaded_vectorizer = joblib.load(vectorizer_path)
        return loaded_model, loaded_vectorizer
    except Exception as e:
        st.error(f"Error loading model or vectorizer. Please ensure files are in '{model_dir}'. Error: {e}")
        st.stop()

loaded_model, loaded_vectorizer = load_resources()


# --- Rest of the Streamlit UI and Logic ---
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Enter a movie review below and get an instant sentiment prediction!")

review_input = st.text_area("Your Movie Review:", height=150, placeholder="e.g., This movie was absolutely fantastic, a true masterpiece!")

if st.button("Analyze Sentiment"):
    if review_input:
        with st.spinner("Analyzing..."):
            cleaned_input = preprocess_text(review_input)
            input_features = loaded_vectorizer.transform([cleaned_input])
            prediction = loaded_model.predict(input_features)
            prediction_proba = loaded_model.predict_proba(input_features)

            sentiment_label = "Positive" if prediction[0] == 1 else "Negative"
            confidence_positive = prediction_proba[0][1] * 100
            confidence_negative = prediction_proba[0][0] * 100

            st.subheader("Prediction Result:")
            if sentiment_label == "Positive":
                st.success(f"Sentiment: **{sentiment_label}** 🎉")
                st.progress(confidence_positive / 100)
            else:
                st.error(f"Sentiment: **{sentiment_label}** 😞")
                st.progress(confidence_negative / 100)

            st.write(f"Confidence: Positive ({confidence_positive:.2f}%), Negative ({confidence_negative:.2f}%)")
            st.markdown("---")
            st.markdown(f"**Cleaned Review (for reference):**")
            st.code(cleaned_input)
    else:
        st.warning("Please enter a movie review to analyze.")

st.markdown("---")
st.markdown("Built with ❤️ using Python, scikit-learn, and Streamlit.")