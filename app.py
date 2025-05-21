import streamlit as st
from summarize import load_pdf_chunks, summarize_chunks

st.set_page_config(page_title="TinyLlama PDF Summarizer", layout="centered")
st.markdown("<h1 style='text-align: center; color: white;'>📄 TinyLlama PDF Summarizer</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Reading PDF and summarizing with TinyLlama..."):
        chunks = load_pdf_chunks(uploaded_file)
        summary = summarize_chunks(chunks)

    st.markdown("### Summary:")
    st.markdown(f"<div style='background-color:#343541; padding:15px; border-radius:8px; color:white; white-space: pre-wrap;'>{summary}</div>", unsafe_allow_html=True)
