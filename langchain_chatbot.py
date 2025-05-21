import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# --- Simple login page ---
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    allowed_users = {
        "Akhila": "1234",
        "Raayan": "5678",
        "Olivia": "9876"
    }
    if st.button("Login"):
        if username in allowed_users and password == allowed_users[username]:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success("Login successful!")
        else:
            st.error("Invalid credentials")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False


if not st.session_state["logged_in"]:
    login()
    st.stop()

# --- Main Chatbot App ---
st.title("LangChain PDF Chatbot (Gemini)")

# --- PDF upload ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file is not None:
    # Save uploaded file to disk for PyPDFLoader
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load and split PDF
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    splits = text_splitter.split_documents(docs)

    # --- Use Gemini for embeddings and LLM ---
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    vectordb = FAISS.from_documents(splits, embeddings)

    llm = ChatGoogleGenerativeAI(
        google_api_key=GOOGLE_API_KEY,
        model="models/gemini-1.5-flash"
    )
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectordb.as_retriever())

    # --- Chat history ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Chat form ---
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Enter your question...", key="user_input")
        process = st.form_submit_button("Ask")
        if process and user_input:
            with st.spinner("Thinking..."):
                answer = qa.run(user_input)
            st.session_state.messages.append(("You", user_input))
            st.session_state.messages.append(("Bot", answer))

    # --- Chat window ---
    st.markdown("<h5>Chat History</h5>", unsafe_allow_html=True)
    for sender, message in st.session_state.messages:
        if sender == "You":
            st.markdown(f"<div style='text-align:right;background:#DCF8C6;padding:8px;border-radius:8px;margin:4px 0'><b>You:</b> {message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align:left;background:#F1F0F0;padding:8px;border-radius:8px;margin:4px 0'><b>Bot:</b> {message}</div>", unsafe_allow_html=True)

    # --- Clear chat ---
    if st.button("Clear Chat"):
        st.session_state.messages = []

    # --- Logout ---
    if st.button("Logout"):
        st.session_state.clear()

    # Clean up temp file
    os.remove("temp.pdf")
else:
    st.info("Please upload a PDF to start chatting.")