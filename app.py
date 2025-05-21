import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load your Google API key from .env file
load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

st.title("Gemini Chatbot")

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
    response = model.generate_content(user_input, generation_config=generation_config)
    # Save user and bot messages
    st.session_state.messages.append(("You", user_input))
    st.session_state.messages.append(("Gemini", response.text))

# Display chat history
for sender, message in st.session_state.messages:
    st.markdown(f"**{sender}:** {message}")