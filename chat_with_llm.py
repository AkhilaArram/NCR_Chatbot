import streamlit as st
import google.generativeai as genai
import PyPDF2
import numpy as np
import faiss
import os

# --- Simple login page (for demo only, not secure for production) ---
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Dictionary of allowed users and their passwords
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
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
st.title("Gemini PDF Chatbot Interface")

def chunk_text(text, chunk_size=1500, overlap=300):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# --- FAISS index directory logic ---
INDEX_DIR = "faiss_indexes"
if not os.path.exists(INDEX_DIR):
    os.makedirs(INDEX_DIR)
INDEX_PATH = os.path.join(INDEX_DIR, "my_index.index")

# PDF upload and FAISS index
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
pdf_text = ""
pdf_chunks = []
faiss_index = None
chunk_embeddings = None

def get_google_embedding(text, task_type):
    return np.array(
        genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type=task_type
        )["embedding"],
        dtype=np.float32
    )

def build_faiss_index(chunks):
    embeddings = [get_google_embedding(chunk, "retrieval_document") for chunk in chunks]
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.stack(embeddings))
    return index, embeddings

if uploaded_file is not None:
    reader = PyPDF2.PdfReader(uploaded_file)
    num_pages = len(reader.pages)
    st.info(f"PDF has {num_pages} pages.")
    for page in reader.pages:
        pdf_text += page.extract_text() or ""
    pdf_chunks = chunk_text(pdf_text, chunk_size=1500, overlap=300)

    # If index exists, load it; else, build and save
    if os.path.exists(INDEX_PATH):
        faiss_index = faiss.read_index(INDEX_PATH)
        st.success("Loaded FAISS index from disk.")
        # You still need to rebuild embeddings for similarity search
        chunk_embeddings = [get_google_embedding(chunk, "retrieval_document") for chunk in pdf_chunks]
    else:
        faiss_index, chunk_embeddings = build_faiss_index(pdf_chunks)
        faiss.write_index(faiss_index, INDEX_PATH)
        st.success("Built and saved FAISS index to disk.")
    

# --- Session state for chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

generation_config = {
    "temperature": 1.0,
    "top_p": 0.9,
    "top_k": 40
}

def get_most_relevant_chunks(question, chunks, index, embeddings, top_k=2):
    question_emb = get_google_embedding(question, "retrieval_query")
    D, I = index.search(np.expand_dims(question_emb, axis=0), top_k)
    return [chunks[i] for i in I[0]], D[0]

# --- User input and Process/Send button ---
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Enter your Prompt...", key="user_input")
    process = st.form_submit_button("Process")

    if process and user_input:
        if pdf_chunks and faiss_index is not None:
            relevant_chunks, distances = get_most_relevant_chunks(
                user_input, pdf_chunks, faiss_index, chunk_embeddings, top_k=2
            )
            l2_threshold = 1.3
            if min(distances) > l2_threshold:
                st.session_state.messages.append(("You", user_input))
                st.session_state.messages.append(("Gemini", "Not relevant question."))
            else:
                context = "\n".join(relevant_chunks)
                prompt = f"PDF Content:\n{context}\n\nUser Question: {user_input}"
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(prompt, generation_config=generation_config)
                st.session_state.messages.append(("You", user_input))
                st.session_state.messages.append(("Gemini", response.text))
        else:
            st.session_state.messages.append(("You", user_input))
            st.session_state.messages.append(("Gemini", "Please upload a PDF to ask questions about its content."))

# --- ChatGPT-like chat window ---
st.markdown("<h5>Chat History</h5>", unsafe_allow_html=True)
for sender, message in st.session_state.messages:
    if sender == "You":
        st.markdown(f"<div style='text-align:right;background:#DCF8C6;padding:8px;border-radius:8px;margin:4px 0'><b>You:</b> {message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align:left;background:#F1F0F0;padding:8px;border-radius:8px;margin:4px 0'><b>Gemini:</b> {message}</div>", unsafe_allow_html=True)

#---clear chat button---
if st.button("Clear Chat"):
    st.session_state.messages = []

# --- Session management: Logout button ---
if st.button("Logout"):
    st.session_state.clear()