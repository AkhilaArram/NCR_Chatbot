# filepath: c:\Users\AKHILA\OneDrive\Desktop\GEAR\NCR_Chatbot\test_gemini_key.py
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    google_api_key=GOOGLE_API_KEY,
    model="models/gemini-1.5-pro-latest"
)
try:
    print(llm.invoke("Hello!"))
    print("API key is valid.")
except Exception as e:
    print("API key is invalid or quota exceeded:", e)