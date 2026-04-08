import streamlit as st
import pandas as pd
import numpy as np
import joblib
from gensim.models import KeyedVectors
import re
from nltk.stem import WordNetLemmatizer

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Kindle Review Sentiment Pro", layout="wide")

# Custom CSS for better looks
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e6e9ef;
        margin-bottom: 10px;
    }
    .metric-label {
        color: #555555; /* Explicit dark gray for labels */
        font-size: 16px;
    }
    .metric-value {
        color: #000000; /* Explicit black for the 84.62% values */
        font-size: 32px;
        font-weight: bold;
    }
    </style>
    <div class="metric-card">
        <div class="metric-label">TF-IDF + LogReg</div>
        <div class="metric-value">84.62%</div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- LOAD ASSETS (Cached for Performance) ---
@st.cache_resource
def load_models():
    # Vectorizers
    bow_vec = joblib.load("bow_vectorizer.pkl")
    tfidf_vec = joblib.load("tfidf_vectorizer.pkl")
    
    # Classifiers
    model_bow = joblib.load("model_bow.pkl")
    model_tfidf = joblib.load("model_tfidf.pkl")
    model_w2v = joblib.load("model_w2v.pkl")
    
    # Word2Vec Vectors
    w2v_wv = KeyedVectors.load("kindle_vectors.kv")
    
    return bow_vec, tfidf_vec, model_bow, model_tfidf, model_w2v, w2v_wv

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Loading all models
try:
    bow_vec, tfidf_vec, m_bow, m_tfidf, m_w2v, w2v_wv = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Make sure all .pkl and .kv files are in the same directory.")

# --- HELPER FUNCTIONS ---
def clean_and_lemmatize(text):
    # Basic cleaning matching your training pipeline
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    # Lemmatize
    words = [lemmatizer.lemmatize(word) for word in text.split()]
    return " ".join(words)

def get_w2v_sum(text, wv):
    tokens = text.split()
    vectors = [wv[word] for word in tokens if word in wv]
    if not vectors:
        return np.zeros(100).reshape(1, -1)
    return np.sum(vectors, axis=0).reshape(1, -1)

# --- SIDEBAR METRICS ---
st.sidebar.title("📊 Model Performance")
st.sidebar.markdown("### Test Accuracy Results")

# Displaying your specific results
st.sidebar.metric("TF-IDF + LogReg", "84.62%")
st.sidebar.metric("BoW + LogReg", "84.00%")
st.sidebar.metric("Word2Vec + LogReg", "79.37%")

st.sidebar.divider()
st.sidebar.write("**Note:** Word2Vec was trained with 100 dimensions to optimize for 8GB RAM.")

# --- PERFORMANCE CHART ---
st.subheader("Model Accuracy Comparison")
results_df = pd.DataFrame({
    'Method': ['BoW', 'TF-IDF', 'Word2Vec'],
    'Accuracy': [0.84, 0.84625, 0.79375]
})

# Use a bar chart to show the comparison
st.bar_chart(data=results_df, x='Method', y='Accuracy')

st.info("""
**Observation:** TF-IDF performed best in this scenario. 
This often happens in sentiment analysis because 'keywords' are very strong signals, 
whereas Word2Vec requires much larger datasets to fully map out word meanings.
""")

# --- MAIN UI ---
st.title("📚 Kindle Review Sentiment Analyzer")
st.write("Compare how different NLP methods interpret your review.")

user_input = st.text_area("Type your Kindle review here:", height=150, placeholder="Example: This book was amazing, I loved every chapter!")

if st.button("Analyze & Compare Results"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        with st.spinner('Computing vectors and predicting...'):
            # Preprocess
            processed_text = clean_and_lemmatize(user_input)
            
            # 1. BoW Prediction
            bow_input = bow_vec.transform([processed_text])
            res_bow = m_bow.predict(bow_input)
            
            # 2. TF-IDF Prediction
            tfidf_input = tfidf_vec.transform([processed_text])
            res_tfidf = m_tfidf.predict(tfidf_input)
            
            # 3. Word2Vec Prediction
            w2v_input = get_w2v_sum(processed_text, w2v_wv)
            res_w2v = m_w2v.predict(w2v_input)

            # --- DISPLAY RESULTS ---
            st.divider()
            col1, col2, col3 = st.columns(3)
            
            def display_result(col, title, prediction):
                sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
                color = "green" if prediction == 1 else "red"
                col.markdown(f"### {title}")
                col.markdown(f"<h2 style='color: {color};'>{sentiment}</h2>",unsafe_allow_html=True)

            display_result(col1, "Bag of Words", res_bow)
            display_result(col2, "TF-IDF", res_tfidf)
            display_result(col3, "Word2Vec", res_w2v)

            # --- COMPARATIVE ANALYSIS ---
            st.divider()
            st.subheader("Why are they different?")
            expander = st.expander("See technical explanation")
            expander.write("""
                * **BoW:** Only cares about the presence and count of words. It can be fooled by sarcasm or word order.
                * **TF-IDF:** Down-weights common words (like 'the', 'book') and highlights unique sentiment words.
                * **Word2Vec:** Understands context. It 'knows' that 'fantastic' and 'excellent' are similar in meaning.
            """)