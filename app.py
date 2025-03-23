import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "  # Ensuring proper spacing between pages
    return text.strip()

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([job_description] + resumes)
    
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    
    cosine_similarities = cosine_similarity(job_description_vector, resume_vectors).flatten()
    return cosine_similarities

# Streamlit app
st.title("AI Resume Screening & Candidate Ranking System")

# Job description input
st.header("Job Description")
job_description = st.text_area("Enter the job description")

# File uploader
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Process resumes and rank them
if uploaded_files and job_description:
    st.header("Ranking Resumes")

    resumes = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)

    # Rank resumes
    scores = rank_resumes(job_description, resumes)

    # Create a results DataFrame
    results = pd.DataFrame({
        "Resume": [file.name for file in uploaded_files],
        "Score": scores
    })

    # Sort by score in descending order
    results = results.sort_values(by="Score", ascending=False)

    # Display the results
    st.write(results)