# -------------------------
# 📁 patent_landscape_dashboard.py
# -------------------------
import streamlit as st
import pandas as pd
import numpy as np
import nltk
import spacy
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import io

# Ensure necessary downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load BERT
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------
# 📌 Utility Functions
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and w.isalpha()]
    return ' '.join(tokens)

def preprocess_dataframe(df):
    df['combined'] = df['Title'].fillna('') + ' ' + df['Abstract'].fillna('') + ' ' + df['Claims'].fillna('')
    df['clean_text'] = df['combined'].apply(clean_text)
    return df

def build_taxonomy_tree():
    st.markdown("### 📚 Define Taxonomy Hierarchy (Tree Structure)")
    taxonomy_tree = []
    with st.expander("Add Taxonomy Tree"):
        with st.form("taxonomy_form"):
            st.write("Add multiple taxonomy paths from root to leaves")
            path_container = st.container()
            num_paths = st.number_input("How many taxonomy paths do you want to define?", min_value=1, max_value=50, value=3)
            for i in range(num_paths):
                path = st.text_input(f"Path {i+1} (separate levels with ' > '):", key=f"path_{i}")
                if path:
                    taxonomy_tree.append(path.strip())
            submitted = st.form_submit_button("Submit Taxonomy Tree")
    return taxonomy_tree

def get_tfidf_features(texts):
    vectorizer = TfidfVectorizer(max_features=300)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix

def get_bert_embeddings(texts):
    return bert_model.encode(texts, show_progress_bar=True)

def ensemble_vectors(tfidf_matrix, bert_matrix):
    tfidf_matrix = normalize(tfidf_matrix)
    bert_matrix = normalize(bert_matrix)
    combined = np.hstack((tfidf_matrix.toarray(), bert_matrix))
    return combined

def compute_probabilities(patent_vecs, taxonomy_vecs, taxonomy_labels):
    similarity = cosine_similarity(patent_vecs, taxonomy_vecs)
    prob_df = pd.DataFrame(similarity, columns=taxonomy_labels)
    return prob_df

# -------------------------
# 🚀 Streamlit App
# -------------------------
st.set_page_config(layout="wide")
st.title("🔍 Patent Landscape Analysis Dashboard")

# Upload File
uploaded_file = st.file_uploader("Upload Patent File (CSV or Excel)", type=['csv', 'xlsx'])
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    required_cols = {'Patent Number', 'Title', 'Abstract', 'Claims'}
    if not required_cols.issubset(set(df.columns)):
        st.error("File must contain columns: Patent Number, Title, Abstract, Claims")
    else:
        st.success("File uploaded successfully")
        st.dataframe(df.head())

        # Domain Input
        domain = st.text_input("Enter the Domain for Landscape Analysis (e.g., Wildfire Sensors)")

        # Taxonomy Tree Input
        taxonomy_tree = build_taxonomy_tree()

        if st.button("Process and Tag Patents") and taxonomy_tree:
            with st.spinner("Preprocessing text and generating embeddings..."):
                df = preprocess_dataframe(df)
                #tfidf_vecs = get_tfidf_features(df['clean_text'])
                #bert_vecs = get_bert_embeddings(df['clean_text'].tolist())
                #patent_vecs = ensemble_vectors(tfidf_vecs, bert_vecs)
                patent_vecs = get_bert_embeddings(df['clean_text'].tolist())

                taxonomy_labels = taxonomy_tree
                taxonomy_vecs = get_bert_embeddings(taxonomy_labels)

                prob_df = compute_probabilities(patent_vecs, taxonomy_vecs, taxonomy_labels)
                final_df = pd.concat([df[['Patent Number', 'Title']], prob_df], axis=1)

            st.success("Tagging Complete")
            st.dataframe(final_df.head(10))
            #st.download_button("Download Results as CSV", data=final_df.to_csv(index=False), file_name="patent_tagging_results.csv")
            csv_data = final_df.to_csv(index=False)
            csv_bytes = io.BytesIO(csv_data.encode('utf-8'))

            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_bytes,
                file_name="patent_tagging_results.csv",
                mime="text/csv"
            )
