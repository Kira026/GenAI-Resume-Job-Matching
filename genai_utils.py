# genai_utils.py
import os
from groq import Groq
from dotenv import load_dotenv

# Load .env explicitly (SAFE for local use)
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

client = Groq(api_key=api_key)


def build_prompt(resume_text, job_text, score):
    return f"""
You are a senior hiring manager.

Match Score (0–10): {score:.2f}

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_text}

Explain:
1. Why this resume matches or does not match
2. Missing or weak skills
3. Concrete improvement suggestions

Use bullet points and be concise.
"""


def get_explanation(prompt):
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are an expert hiring manager."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content
