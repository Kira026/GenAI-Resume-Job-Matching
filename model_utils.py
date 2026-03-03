# model_utils.py
import re
import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load artifacts
tfidf = joblib.load("tfidf.pkl")
model = joblib.load("ridge_model.pkl")
scaler = joblib.load("scaler.pkl")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

def compute_match_score(resume_text, job_text):
    resume_clean = preprocess_text(resume_text)
    job_clean = preprocess_text(job_text)

    resume_len = len(resume_clean.split())
    job_len = len(job_clean.split())

    tfidf_r = tfidf.transform([resume_clean])
    tfidf_j = tfidf.transform([job_clean])
    tfidf_sim = cosine_similarity(tfidf_r, tfidf_j)[0][0]

    r_emb = embedder.encode([resume_clean])
    j_emb = embedder.encode([job_clean])
    semantic_sim = cosine_similarity(r_emb, j_emb)[0][0]

    X = np.array([[tfidf_sim, semantic_sim, resume_len, job_len]])

    score_scaled = model.predict(X)[0]
    score = scaler.inverse_transform([[score_scaled]])[0][0]

    return score, {
        "TF-IDF Similarity": round(tfidf_sim, 3),
        "Semantic Similarity": round(semantic_sim, 3),
        "Resume Length": resume_len,
        "Job Length": job_len
    }
