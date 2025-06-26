import streamlit as st
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, BaseMessage
from typing import TypedDict, Annotated, Sequence
from langgraph.graph.message import add_messages

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it in your .env file for local use, or in Streamlit secrets for deployment.")
    st.stop()

# --- State and Graph Definition ---

class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)

# LangGraph Chat Node
def chat_node(state: GraphState) -> dict:
    """Invokes the LLM with the entire conversation history."""
    # The LLM now gets the full context of the conversation
    response = llm.invoke(state["messages"])
    # We return only the new message to be added to the state
    return {"messages": [response]}

# Build the graph
graph_builder = StateGraph(GraphState)
graph_builder.add_node("chat", chat_node)
graph_builder.set_entry_point("chat")
graph_builder.add_edge("chat", END)
graph = graph_builder.compile()

# --- Streamlit UI ---
st.title("Chatbot")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages from session state
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)
        
# Handle user input
if user_input := st.chat_input("Ask a question:"):
    # Add user message to session state and display it
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("human"):
        st.markdown(user_input)

    # Invoke the graph with the full conversation history
    with st.spinner("Thinking..."):
        final_state = graph.invoke({"messages": st.session_state.messages})
        bot_reply_message = final_state["messages"][-1]

    # Add bot's response to session state and display it
    st.session_state.messages.append(bot_reply_message)
    with st.chat_message(bot_reply_message.type):
        st.markdown(bot_reply_message.content)

# Display feedback options for the last bot message
if st.session_state.messages and st.session_state.messages[-1].type != "human":
    st.write("Was this response helpful?")
    
    col1, col2, _ = st.columns([1, 1, 10])
    feedback_key = f"feedback_{len(st.session_state.messages)}"

    if col1.button("👍 Yes", key=f"yes_{feedback_key}"):
        st.success("Thank you for the positive feedback!")
    if col2.button("👎 No", key=f"no_{feedback_key}"):
        st.warning("Thank you for your feedback! We'll try to improve.")
