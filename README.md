🧠 GenAI-Powered Resume ↔ Job Matching System
🚀 Overview

This project is an end-to-end AI system that intelligently matches resumes with job descriptions using a hybrid Machine Learning and Generative AI architecture.

It simulates how modern Applicant Tracking Systems (ATS) score resumes while also providing human-like explanations and improvement suggestions.

🏗 Architecture

Resume (Text / PDF)
→ PDF Parsing + OCR
→ Text Cleaning & Normalization
→ Feature Engineering
→ Hybrid ML Model (TF-IDF + Semantic Embeddings)
→ Match Score Prediction
→ LLM-Based Explanation
→ Streamlit Web App

🔍 Key Features

Resume input (Text + PDF)

OCR support for scanned resumes

TF-IDF based keyword similarity

Sentence Transformer semantic similarity

Ridge Regression scoring model

Hybrid ML + LLM architecture

GenAI explanation with improvement suggestions

Interactive Streamlit UI

📊 Model Performance

Baseline Model:

RMSE: 1.74

R²: 0.18

Hybrid Model:

RMSE: 1.71

R²: 0.21

The hybrid approach improved predictive performance by incorporating semantic similarity alongside lexical matching.

🛠 Tech Stack

Python

Pandas

Scikit-Learn

TF-IDF Vectorization

Sentence Transformers

Ridge Regression

PyMuPDF

Tesseract OCR

Streamlit

Groq LLM API
