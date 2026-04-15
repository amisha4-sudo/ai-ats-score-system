import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("AI Resume Analyzer + ATS Score")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
job_desc = st.text_area("Paste Job Description")

def extract_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

if uploaded_file and job_desc:
    resume_text = extract_text(uploaded_file)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_desc])

    score = (vectors * vectors.T).A[0,1] * 100

    st.subheader("ATS Score")
    st.write(f"{round(score,2)} %")

    resume_words = set(resume_text.lower().split())
    jd_words = set(job_desc.lower().split())

    missing = jd_words - resume_words

    st.subheader("Missing Keywords")
    st.write(list(missing)[:20])

    st.subheader("Suggestions")
    st.write("Add missing keywords and relevant skills.")
