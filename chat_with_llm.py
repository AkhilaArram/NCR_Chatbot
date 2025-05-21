import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import PyPDF2

# Load your Google API key from .env file
load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

st.title("Gemini Chatbot")

# PDF upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
pdf_text = ""
if uploaded_file is not None:
    reader = PyPDF2.PdfReader(uploaded_file)
    for page in reader.pages:
        pdf_text += page.extract_text() or ""

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("You:", "")

generation_config = {
    "temperature": 1.0,
    "top_p": 0.9,
    "top_k": 40
}

if user_input:
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = user_input
    if pdf_text:
        # Warn if PDF is very large
        if len(pdf_text) > 12000:
            st.warning("PDF content is very large. Only the first part will be used.")
            pdf_text = pdf_text[:12000]
        prompt = f"PDF Content:\n{pdf_text}\n\nUser Question: {user_input}"
    response = model.generate_content(prompt, generation_config=generation_config)
    st.session_state.messages.append(("You", user_input))
    st.session_state.messages.append(("Gemini", response.text))

# Display chat history
for sender, message in st.session_state.messages:
    st.markdown(f"**{sender}:** {message}")