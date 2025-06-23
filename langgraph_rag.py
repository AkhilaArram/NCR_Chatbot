from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader # Changed PyPDFLoader to UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google.api_core.exceptions import ResourceExhausted # Import the exception
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found. Please set it in your .env file or environment variables.")
    exit()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature = 0) 

#  Embedding Model  
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

pdf_path = r"C:\Users\AKHILA\OneDrive\Desktop\forest_eng.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = UnstructuredPDFLoader(pdf_path, mode="elements", strategy="hi_res", pdf_image_converter="poppler")
# Checks if the PDF is there
try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and broken into {len(pages)} elements/chunks by UnstructuredPDFLoader.")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

# Chunking Process
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

pages_split = text_splitter.split_documents(pages) # We now apply this to our pages

try:
    # Create the FAISS vector store
    vectorstore = FAISS.from_documents(
        documents=pages_split,
        embedding=embeddings
    )
    print(f"Created FAISS vector store in memory!")
    
except Exception as e:
    print(f"Error setting up FAISS: {str(e)}")
    raise

# retriever 
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} 
)

@tool

def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the  forest_eng.pdf PDF document.
    """

    docs = retriever.invoke(query)
    print(f"\nDEBUG: Retriever found {len(docs)} documents for query: '{query}'")

    if not docs:
        print("DEBUG: No documents found by retriever for this query.")
        return "Based on the query, no relevant information was found in the 'forest_eng.pdf' document."
    
    results = []
    print("DEBUG: Snippets of retrieved documents content:")
    for i, doc in enumerate(docs):
        # UnstructuredPDFLoader often puts source and page_number in metadata
        source_info = doc.metadata.get('filename', 'Unknown Filename')
        page_info = doc.metadata.get('page_number', 'Unknown Page')
        element_type = doc.metadata.get('category', 'Unknown Element')
        print(f"--- DOC {i+1} (File: {source_info}, Page: {page_info}, Type: {element_type}) ---")
        # --- MODIFICATION START: Print full content for the first document ---
        if i < 3: # Let's inspect the top 3 most relevant documents fully
            print(f"FULL CONTENT of DOC {i+1}:\n{doc.page_content}\n")
        else: # Print snippets for others
            print(f"{doc.page_content[:400]}...")
        # --- MODIFICATION END ---
        print("--- END DOC ---")
        # MODIFICATION: Send only a snippet of each document to the LLM
        snippet_length = 750 # Characters
        results.append(f"Document {i+1} (Source: {source_info}, Page: {page_info}, Type: {element_type}):\n{doc.page_content[:snippet_length]}...")
    return "\n\n".join(results)

tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


system_prompt = """
You are a highly specialized AI assistant. Your ONLY function is to provide information based EXCLUSIVELY on the content of the 'forest_eng.pdf' document.

**Core Instruction for ALL User Inputs:**
1.  **Interpret User Input as a Search Task:** Regardless of how the user phrases their input (e.g., as a direct question "What is X?", a command "Explain X", a statement "Tell me about X", or even a conversational fragment), your primary goal is to extract the core subject or topic the user is interested in from 'forest_eng.pdf'.
2.  **Formulate an Effective Search Query:** Based on the identified core subject/topic, formulate an effective search query. This query should consist of key terms and concepts. For example, if the user says "Explain about the different types of trees in the document", a good search query for the `retriever_tool` would be "types of trees" or "tree species". Do NOT include conversational phrases like "explain about" or "tell me" in the query you pass to the tool.
3.  **Mandatory Tool Use:** You MUST use the 'retriever_tool' with this formulated search query to find relevant information from the 'forest_eng.pdf' document. This is your FIRST and ONLY action to gather information.

**Response Protocol:**
*   **Information Found:** If the 'retriever_tool' provides relevant document excerpts:
    *   Synthesize your answer based SOLELY on these provided excerpts.
    *   Clearly state that your answer is derived from the 'forest_eng.pdf' document.
    *   If possible, cite specific parts of the documents (e.g., page numbers, section titles if available from the tool's output).
*   **No Information Found:** If the 'retriever_tool' returns a message indicating no relevant information was found (e.g., "Based on the query, no relevant information was found in the 'forest_eng.pdf' document."):
    *   You MUST respond with the exact phrase: "I can only answer questions based on the content of the 'forest_eng.pdf' document, and that information was not found."
    *   Do NOT attempt to answer using external knowledge or make assumptions.

**Strict Prohibitions:**
*   **No External Knowledge:** Under NO circumstances should you use any external knowledge, personal opinions, or information not directly retrieved from 'forest_eng.pdf' by the 'retriever_tool'.
*   **No General Conversation (Beyond Initial Search):** Do not answer general conversational questions (e.g., "hello," "how are you," "what is your name?") with anything other than the process outlined above (i.e., attempt to search the PDF for relevance, and if none, provide the "no information found" response).
*   **No Assumptions:** Do not make assumptions or infer information beyond what is explicitly stated in the retrieved document excerpts.

Your entire purpose is to be a focused, factual interface to the 'forest_eng.pdf' document. Do not deviate from these instructions.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools} # Creating a dictionary of tools

# LLM Agent Node
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages': [message]}


# Retriever Agent Node
def take_action(state: AgentState) -> dict:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result))) # type: ignore

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()
# Visualization of Graph
# rag_agent.get_graph().draw_mermaid_png(output_file_path="rag_visualization.png")
# print("Graph visualization saved to rag_visualization.png")

def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        try:
            messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

            result = rag_agent.invoke({"messages": messages})
            
            print("\n=== ANSWER ===")
            print(result['messages'][-1].content)
        except ResourceExhausted as e:
            print("\n=== API ERROR ===")
            print("The API rate limit has been exceeded. Please wait a moment and try again, or consider upgrading your API plan.")
            print(f"Details: {e}")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
running_agent()