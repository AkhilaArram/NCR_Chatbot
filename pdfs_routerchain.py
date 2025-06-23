# Gemini PDF Chatbot with RouterChain, History, and Summarization
import os
import glob
import re # For robust JSON extraction
import json
import time
from fpdf import FPDF
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfMerger
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.router.llm_router import RouterOutputParser # Keep for inheritance
from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException # For custom parser
from langchain_core.runnables import RunnableLambda, RunnableBranch
from typing import Dict, Any

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it in your .env file.")
    st.stop()

PDF_DIR = "pdfs"
COMBINED_PDF = "combined.pdf"
CHAT_HISTORY_FILE = "chat_history.json"
SUMMARY_FILE = "summary.txt"

# Initialize generative model
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    llminstance = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="models/gemini-1.5-flash", temperature=0.7)
    # Dedicated LLM for routing with low temperature for deterministic JSON output
    router_llm = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="models/gemini-1.5-flash", temperature=0.0)
except Exception as e:
    st.error(f"Failed to initialize Google Generative AI: {e}")
    st.stop()
    
# Create PDFs if not exists
topics = ["forest", "beach", "sea", "trees", "flowers", "mountains", "desert", "river", "lake", "rainforest",
          "savanna", "tundra", "volcano", "island", "canyon", "waterfall", "meadow", "valley", "swamp", "reef",
          "prairie", "glacier", "bay", "lagoon", "delta", "grove", "orchard", "jungle", "cliff", "plateau",
          "hill", "plain", "dune", "marsh", "steppe", "woodland", "mangrove", "oasis", "peninsula", "cape",
          "gulf", "fjord", "atoll", "archipelago", "shoal", "moor", "badlands", "rainbow", "aurora", "geyser"]

os.makedirs(PDF_DIR, exist_ok=True)
pdfs_generated_this_session = 0
if not all(os.path.exists(os.path.join(PDF_DIR, f"{topic}.pdf")) for topic in topics):
    st.info("Generating PDFs for the first time or if some are missing. This may take a while...")
    for i, topic in enumerate(topics):
        pdf_path = os.path.join(PDF_DIR, f"{topic}.pdf")
        if not os.path.exists(pdf_path):
            if pdfs_generated_this_session > 0 and pdfs_generated_this_session % 10 == 0:
                st.info(f"Generated {pdfs_generated_this_session} PDFs, pausing for a bit to respect API rate limits...")
                time.sleep(10)

            try:
                model_gen = genai.GenerativeModel('gemini-1.5-flash')
                response = model_gen.generate_content(f"Write an article about {topic}.")
                content = response.text
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                safe_content = content.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 10, safe_content)
                pdf.output(pdf_path)
                pdfs_generated_this_session += 1
                st.write(f"Generated PDF for {topic} ({i+1}/{len(topics)})")
                time.sleep(3)
            except Exception as e:
                st.error(f"Error generating PDF for {topic}: {e}")
                time.sleep(5)
                continue
    if pdfs_generated_this_session > 0:
        st.success("PDF generation complete.")
    else:
        st.info("All PDFs already exist or no new PDFs were generated.")

# Merge PDFs
pdf_files = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
if pdf_files:
    merger = PdfMerger()
    for f_path in pdf_files:
        try:
            merger.append(f_path)
        except Exception as e:
            st.warning(f"Could not append {f_path} to merged PDF: {e}")
    try:
        merger.write(COMBINED_PDF)
        merger.close()
    except Exception as e:
        st.error(f"Could not write merged PDF {COMBINED_PDF}: {e}")
else:
    st.error(f"No PDF files found in '{PDF_DIR}'. Cannot proceed without PDFs. Please ensure PDFs are generated.")
    st.stop()

# Load and embed docs
all_docs_from_pdfs = []
for file_path_load in pdf_files:
    try:
        loader = PyPDFLoader(file_path_load)
        all_docs_from_pdfs.extend(loader.load())
    except Exception as e:
        st.warning(f"Could not load PDF {file_path_load}: {e}")

if not all_docs_from_pdfs:
    st.error("No documents could be loaded from the PDFs. Please check PDF content and integrity.")
    st.stop()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(all_docs_from_pdfs)

try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vectordb = FAISS.from_documents(splits, embeddings)
except Exception as e:
    st.error(f"Failed to create FAISS vector store for PDF content: {e}")
    st.stop()

NUM_PDFS_TO_SUMMARIZE = 1 # Changed to summarize only the first PDF
docs_for_summary = []
# pdf_files is already populated with all PDF paths from PDF_DIR, sorted.
pdf_files_for_summary = pdf_files[:NUM_PDFS_TO_SUMMARIZE]

if not pdf_files_for_summary:
    st.warning(f"No PDF files found to select the first {NUM_PDFS_TO_SUMMARIZE} for summarization from {PDF_DIR}.")
else:
    st.info(f"Loading documents from the first {len(pdf_files_for_summary)} PDFs for summarization: {', '.join(os.path.basename(p) for p in pdf_files_for_summary)}")
    for file_path_summary in pdf_files_for_summary:
        try:
            loader_summary = PyPDFLoader(file_path_summary)
        
            docs_for_summary.extend(loader_summary.load())
        except Exception as e:
            st.warning(f"Could not load PDF {os.path.basename(file_path_summary)} for summarization: {e}")

# --- Summarization (using docs_for_summary) ---
summarizer = load_summarize_chain(llminstance, chain_type="map_reduce")
docs_to_summarize = docs_for_summary # Use documents from the first N PDFs
summary = "Could not generate summary."

if os.path.exists(SUMMARY_FILE):
    st.info(f"Loading existing summary from {SUMMARY_FILE}...")
    try:
        with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
            summary = f.read()
    except Exception as e:
        st.error(f"Error loading summary from {SUMMARY_FILE}: {e}")
        summary = "Failed to load existing summary."
else:
    if docs_to_summarize:
        try:
            st.info(f"Generating summary from {len(docs_to_summarize)} document sections from the first {len(pdf_files_for_summary)} PDFs. This may take API calls...")
            summary_result = summarizer.invoke({"input_documents": docs_to_summarize})
            summary = summary_result.get("output_text", "Summary not found in chain output.")
            with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
                f.write(summary)
            st.info("Summary generated and saved to summary.txt.")
        except Exception as e:
            st.error(f"Error during summarization: {e}")
    elif not os.path.exists(SUMMARY_FILE): # Only warn if file doesn't exist and no docs were prepared for summary
        st.warning("No documents available to summarize.")

summary_doc = Document(page_content=summary)
try:
    summary_vectordb = FAISS.from_documents([summary_doc], embeddings)
except Exception as e:
    st.error(f"Failed to create FAISS vector store for summary: {e}")
    summary_vectordb = None

# --- Chat History VectorDB and Chains Setup ---
# This needs `embeddings` (already global), `llminstance` (global), and `history_qa_prompt`.

HISTORY_QA_TEMPLATE = """
You are a helpful AI assistant. You have access to previous parts of the conversation history.
Your SOLE purpose when answering the user's current question is to use the "Retrieved Conversation History" provided below.
Do NOT use your general knowledge or any other information source to answer about the topic itself if it's not in the history.

Instructions for answering:
1.  **Analyze the User's Current Question:**
    *   Identify the main subject of the user's question about past discussions. Let's call this the Query_Subject.
    *   Determine if the question is simple (e.g., "Did we discuss Query_Subject?") or compound (e.g., "Did we discuss Query_Subject? If so, explain Query_Subject.").
    *   Also, handle general queries like "What did we talk about?".

2.  *   For Specific Topic Queries (e.g., "Did we discuss [Query_Subject]?"):**
    *   Examine the "Retrieved Conversation History" ({context}).
    *   Determine if any Q&A pairs in the {context} are directly about the Query_Subject or a very closely related aspect or sub-topic of the Query_Subject.
        (For example, if Query_Subject is "volcanoes", a "very closely related aspect" could be "volcano eruptions" or "types of volcanoes". If Query_Subject is "trees", a related aspect could be "deforestation" or "pine trees".)
    *   If such relevant Q&A pairs are found (let's call the topic found in history History_Topic):
        *   Acknowledge this. If History_Topic is the same as Query_Subject, you can say: "Yes, we previously discussed [Query_Subject]."
        *   If History_Topic is slightly different but very closely related, state what was discussed. For example: "Yes, we touched upon a related topic. Our conversation included discussion about [History_Topic]."
        *   If the user's question was simple (e.g., "Did we discuss volcanoes?"), and you found "volcano eruptions" as the History_Topic, you might add: "Were you referring to our discussion on volcano eruptions, or do you have a more general question about volcanoes?"
    *   If no Q&A pairs directly about Query_Subject or a very closely related aspect are found in {context} (or {context} is empty/placeholder):
        Respond: "No, it doesn't look like we've specifically discussed [Query_Subject] in our current conversation. If you'd like to know if I have information on [Query_Subject] from my documents, you can ask a direct question such as 'Tell me about [Query_Subject]' or 'What do the documents say about [Query_Subject]?'."

3.  **Address the "Conditional Explanation" Part of the Query (if present):**
    *   This applies if step 2 determined that Query_Subject (or a closely related History_Topic) *was* discussed.
    *   **Scenario A: "IF SO, explain [Query_Subject]"**:
        a.  Look for the specific explanation of Query_Subject (or History_Topic if that's what was found and is relevant to the explanation request) in the "Retrieved Conversation History".
        b.  If the explanation is found in the history: Provide it directly from the history. (e.g., "Yes, we previously discussed mountains. Regarding their explanation, the history shows: [explanation from history].")
        c.  If an explanation for the specific Query_Subject (or the relevant History_Topic) is NOT found in the history (even if the topic was mentioned):
            Acknowledge what was discussed (e.g., "Yes, we discussed [History_Topic]."). Then add: "...However, a specific explanation for [Query_Subject or History_Topic, as appropriate] is not in our retrieved conversation. If you'd like an explanation from the documents, please ask this as a new, direct question (e.g., 'Explain [Query_Subject]')."    
             Do NOT provide an explanation from general knowledge.

    *   **Scenario B: "IF NOT, explain [Query_Subject]"** (and Query_Subject was *not* discussed according to step 2):
        a.  Start with the response from step 2 for "[Query_Subject] was NOT discussed". Then append: "Since we haven't discussed it, I cannot provide an explanation based on our past conversation. As mentioned, if you would like an explanation of [Query_Subject] from the documents, please ask a new, direct question (e.g., 'Explain [Query_Subject]')."
          
        Do NOT provide an explanation from general knowledge or PDFs yourself. Your role is to report on history and guide the user.

    *   **Scenario C: Direct question routed to history (e.g., user asks "Explain [Query_Subject]" and router thinks it's a follow-up):**
        a.  Check if an explanation for Query_Subject is in the "Retrieved Conversation History".
        b.  If found: Provide it. (e.g., "Regarding X, our previous conversation includes: [content from history].")
        c.  If not found: State: "I don't have information about [Query_Subject] in our current conversation history. If you'd like this information from the documents, please ask this as a new, direct question."
4.  **For General Queries about Past Discussions (e.g., "What topics/questions did we discuss?"):**
    *   Examine the "Retrieved Conversation History" ({context}).
    *   If the {context} is empty, contains only a placeholder (e.g., "Initial placeholder for chat history vector store.", "Chat history is currently unavailable."), or does not contain any actual Q&A pairs from the current session:
        Respond: "Based on our current conversation, it seems we haven't discussed any specific topics yet, or the history is not detailed enough to list them."
    *   Otherwise (if {context} contains actual Q&A pairs from the current session):
        List the main questions or topics evident from the "Previous Question:" parts of the {context}. For example: "Based on our current conversation, we've touched upon questions like: '[Previous Question 1]', '[Previous Question 2]'." or "So far in our conversation, we've discussed topics related to: [topic from Q1], [topic from Q2]."
        Be literal to the content of the retrieved Q&A pairs. Do not invent topics.

5.  **Strict Adherence to Provided Context and Role:**
    *   If the user's question is only "Did we discuss X?", your answer should be the full response formulated in step 2.
    *   Do not add any extra information or explanations unless explicitly requested as part of a conditional clause AND that information is present in the retrieved history.
    *   Your response should clearly indicate what information comes from history and when information is lacking in history.

    *   Remember, your primary function here is to report on the contents (or absence of contents) of the "Retrieved Conversation History" and guide the user on how to get information from documents if it's not in the history. You do not access documents or general knowledge yourself in this role.
Retrieved Conversation History:
{context}

User's Current Question:
{question}

Answer:"""
history_qa_prompt = PromptTemplate(template=HISTORY_QA_TEMPLATE, input_variables=["context", "question"])

if "history_vectordb" not in st.session_state:
    try:
        # Initialize with a placeholder document.
        placeholder_doc = Document(page_content="Initial placeholder for chat history vector store.")
        st.session_state.history_vectordb = FAISS.from_documents([placeholder_doc], embeddings)

        # Load existing history from file and add to this instance
        chat_history_from_file_for_vdb = []
        if os.path.exists(CHAT_HISTORY_FILE):
            try:
                with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f_vdb:
                    chat_history_from_file_for_vdb = json.load(f_vdb)
            except Exception as e_vdb_load:
                st.warning(f"Error loading chat history from {CHAT_HISTORY_FILE} for VDB init: {e_vdb_load}")

        initial_history_docs_to_add = [
            Document(page_content=f"Previous Question: {item['question']}\nPrevious Answer: {item['text']}")
            for item in chat_history_from_file_for_vdb
            if item.get("sender") == "Gemini" and item.get("question") and item.get("text")
        ]

        if initial_history_docs_to_add:
            st.session_state.history_vectordb.add_documents(initial_history_docs_to_add)

    except Exception as e_hist_vdb_init:
        st.error(f"Fatal error initializing history_vectordb: {e_hist_vdb_init}")
        st.session_state.history_vectordb = None

# Define static chains first
qa_chain = RetrievalQA.from_chain_type(llm=llminstance, retriever=vectordb.as_retriever(), return_source_documents=True)
summary_chain = RetrievalQA.from_chain_type(llm=llminstance, retriever=summary_vectordb.as_retriever(), return_source_documents=True) if summary_vectordb else qa_chain

# Define history_chain using the session_state vector store
if st.session_state.get("history_vectordb"):
    history_retriever = st.session_state.history_vectordb.as_retriever(search_kwargs={"k": 2}) # Increased k to 2
    history_chain = RetrievalQA.from_chain_type(
        llm=llminstance,
        retriever=history_retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": history_qa_prompt},
        return_source_documents=True
    )
else:
    st.warning("history_vectordb is not available. Chat history features will be degraded.")
    class DummyRetriever(BaseRetriever): # Simple fallback retriever
        def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]: return [Document(page_content="Chat history is currently unavailable.")]
        async def _aget_relevant_documents(self, query: str, **kwargs) -> list[Document]: return [Document(page_content="Chat history is currently unavailable.")]
    history_chain = RetrievalQA.from_chain_type(llm=llminstance, retriever=DummyRetriever(), chain_type="stuff", chain_type_kwargs={"prompt": history_qa_prompt}, return_source_documents=True)

# --- Chain to directly get the text of the actual previous user question ---
def retrieve_previous_user_question_text(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieves the text of the actual previous question asked by the user.
    It uses the global `chat_history` list, which reflects the conversation
    state *before* the current user query (e.g., "what was my last question?") is processed.
    The input_dict (user's current query) is not directly used here as we fetch from prior history.
    """
    # chat_history is a global list, accessible here.

    last_q_text = "You haven't asked any questions before this one in this session."

    if chat_history: # Check if the list is not empty
        # The last item in chat_history (either a "You" or "Gemini" entry)
        # should contain the 'question' field with the text of the user's last utterance
        # before the current query being processed.
        last_entry = chat_history[-1]
        if isinstance(last_entry, dict) and "question" in last_entry:
            last_q_text = last_entry["question"]
        else:
            # This case is unlikely if chat_history is populated correctly.
            last_q_text = "Could not retrieve the last question due to unexpected history format."
    return {"result": f"The last question you asked was: \"{last_q_text}\""} # Mimic RetrievalQA output structure
last_question_direct_chain = RunnableLambda(retrieve_previous_user_question_text)

destination_info = [
    {
        "name": "pdf_content_qa",
        "description": "Good for answering questions about the specific content of the uploaded PDF documents. Use this for factual queries based on the provided texts.",
        "chain": qa_chain,
    },
    {
        "name": "chat_history_qa",
        "description": "Good for answering questions about what has been discussed previously in this conversation. Use this if the question refers to past interactions, is a follow-up, or is a repeat of a previous question.",
        "chain": history_chain,
    },
    {
        "name": "document_summary_qa",
        "description": "Good for answering questions about the overall summary of the PDF documents. Use this for high-level overview questions.",
        "chain": summary_chain,
    },
    {
        "name": "get_last_user_question",
        "description": "Use this if the user is asking specifically 'what was the last question I asked', 'what did I just ask', 'what was my previous query', or similar direct inquiries about their immediately preceding utterance. Do not use this for follow-up questions on a topic, only for recalling the literal previous question text.",
        "chain": last_question_direct_chain,
    }
]
# KNOWN_DESTINATION_NAMES and router setup will now use the correctly defined history_chain

KNOWN_DESTINATION_NAMES = [d["name"] for d in destination_info]

CUSTOM_ROUTER_TEMPLATE_STRING = """
Given the user's query and a list of available destinations, select the most appropriate destination.
The available destinations are:
{destinations}

User Query: {input}

<< FORMATTING >>
Respond with ONLY a JSON object. Do not include any other text, explanations, or markdown code block syntax.
The JSON object must conform to the following schema:
{{
    "destination": "string_value_representing_destination_name",
    "next_inputs": "string_value_of_original_or_rephrased_user_query"
}}

"""
# --- Custom Sanitizing Router Output Parser ---
class SanitizingRouterOutputParser(RouterOutputParser):
    def parse(self, text: str) -> Dict[str, Any]:
        print(f"\n[ROUTER_DEBUG] Raw text from LLM for routing: >>>{text}<<<")
        json_text_for_error_reporting = text # Store original text for error reporting
        try:
            # Attempt to extract JSON block more robustly
            match_md = re.search(r"```json\s*([\s\S]+?)\s*```", text, re.DOTALL)
            if match_md:
                json_text = match_md.group(1)
            else:
                # If no markdown fence, try to find a JSON object starting with { and ending with }
                match_obj = re.search(r"^\s*({[\s\S]*})\s*$", text, re.DOTALL)
                if match_obj:
                    json_text = match_obj.group(1)
                else:
                    # Fallback: strip the whole text and hope it's JSON.
                    json_text = text.strip()

            json_text_for_error_reporting = json_text # Update for more specific error context
            json_text = json_text.strip() # Clean the extracted/stripped text
            print(f"[ROUTER_DEBUG] Attempting to parse JSON: >>>{json_text}<<<")

            if not (json_text.startswith("{") and json_text.endswith("}")):
                raise json.JSONDecodeError("Extracted text does not appear to be a JSON object.", json_text, 0)

            parsed_json = json.loads(json_text)

            sanitized_response = {}
            for key, value in parsed_json.items():
                sanitized_key = key.strip() # Strip leading/trailing whitespace from the key string
                # Additionally, remove potential surrounding quotes from the key itself
                if len(sanitized_key) > 1: # Ensure key is not empty or a single quote
                    if sanitized_key.startswith('"') and sanitized_key.endswith('"'):
                        sanitized_key = sanitized_key[1:-1]
                    elif sanitized_key.startswith("'") and sanitized_key.endswith("'"):
                        sanitized_key = sanitized_key[1:-1]
                sanitized_response[sanitized_key] = value

            # Now validate the SANITIZED response
            if "destination" not in sanitized_response:
                raise OutputParserException(
                    f"Sanitized output missing 'destination' key. Original text: '{text[:500]}...'. Attempted to parse: '{json_text_for_error_reporting[:500]}...'. Sanitized response keys: {list(sanitized_response.keys())}")
            if "next_inputs" not in sanitized_response:
                raise OutputParserException(
                    f"Sanitized output missing 'next_inputs' key. Original text: '{text[:500]}...'. Attempted to parse: '{json_text_for_error_reporting[:500]}...'. Sanitized response keys: {list(sanitized_response.keys())}")

            # --- Clean and validate the 'destination' VALUE ---
            raw_destination_name = sanitized_response["destination"]
            if not isinstance(raw_destination_name, str):
                raise OutputParserException(
                    f"Value for 'destination' key from LLM should be a string, but got {type(raw_destination_name)}. Value: {raw_destination_name}")

            cleaned_destination_value = raw_destination_name.strip()
            if len(cleaned_destination_value) > 1: # Remove potential surrounding quotes from the value itself
                if cleaned_destination_value.startswith('"') and cleaned_destination_value.endswith('"'):
                    cleaned_destination_value = cleaned_destination_value[1:-1]
                elif cleaned_destination_value.startswith("'") and cleaned_destination_value.endswith("'"):
                    cleaned_destination_value = cleaned_destination_value[1:-1]

            if cleaned_destination_value not in KNOWN_DESTINATION_NAMES:
                raise OutputParserException(
                    f"LLM returned an invalid or unknown destination name: '{cleaned_destination_value}'. "
                    f"Original value from LLM: '{raw_destination_name}'. Valid destinations are: {KNOWN_DESTINATION_NAMES}")
            # --- End cleaning and validation of 'destination' VALUE ---

            # Ensure next_inputs from LLM is a string as per the prompt's instruction
            raw_next_inputs = sanitized_response["next_inputs"]
            if not isinstance(raw_next_inputs, str):
                raise OutputParserException(
                    f"'next_inputs' from LLM should be a string but got {type(raw_next_inputs)}. Value: {raw_next_inputs}")

            # Format next_inputs for the destination chain (e.g., RetrievalQA expects "query")
            formatted_next_inputs = {"query": raw_next_inputs}

            # Prepare the result dictionary using the cleaned destination value
            result_dict = {
                "destination": cleaned_destination_value,
                "next_inputs": formatted_next_inputs
            }

            print(f"[ROUTER_DEBUG] Successfully parsed and sanitized router output: {result_dict}\n")
            return result_dict
        except json.JSONDecodeError as e:
            print(f"[ROUTER_ERROR] JSONDecodeError during routing: {e}. Attempted to parse: '{json_text_for_error_reporting[:500]}...'")
            raise OutputParserException(f"Failed to decode LLM output as JSON. Attempted to parse: '{json_text_for_error_reporting[:500]}...'. Original text: '{text[:500]}...'. Error: {e}") from e
        except Exception as e:
            print(f"[ROUTER_ERROR] Generic exception during routing: {e}. Original text: '{text[:500]}...'")
            raise OutputParserException(f"Failed to parse or sanitize LLM output. Original text: '{text[:500]}...'. Error: {e}") from e

router_parser = SanitizingRouterOutputParser()
destinations_str = "\n".join([f"{d['name']}: {d['description']}" for d in destination_info])

# Router prompt template
router_prompt_template_obj = PromptTemplate(
    template=CUSTOM_ROUTER_TEMPLATE_STRING,
    input_variables=["input", "destinations"],
    # output_parser is removed here, will be piped in LCEL
)

# --- LCEL Router Chain Construction ---
# This lambda function takes the input dictionary (which contains the user's "input")
# and adds the "destinations" string to it, so the prompt template can be formatted.
# destinations_str is captured from the outer scope.
def add_destinations_to_router_input(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {**input_dict, "destinations": destinations_str}

lcel_router_chain = (
    RunnableLambda(add_destinations_to_router_input)
    | router_prompt_template_obj
    | router_llm
    | router_parser
)
# --- End LCEL Router Chain Construction ---
# --- RunnableBranch for LCEL-native routing ---
# Helper lambda to extract the 'next_inputs' (which is {"query": "..."})
# for the destination chains, as they expect this format.
extract_next_inputs = RunnableLambda(lambda x: x["next_inputs"])

# Define the chains for each branch by piping the extracted inputs to the respective QA chain
pdf_branch_chain = extract_next_inputs | qa_chain
history_branch_chain = extract_next_inputs | history_chain
summary_branch_chain = extract_next_inputs | summary_chain
direct_last_q_branch_chain = last_question_direct_chain # This chain doesn't need extract_next_inputs as it works differently
# Default chain if no other branch is matched (e.g., router outputs an unexpected destination)
# This also needs to process the 'next_inputs'
default_branch_chain_for_routing = extract_next_inputs | qa_chain

# Create the RunnableBranch
# Each branch is a tuple: (condition_lambda, chain_to_run_if_true)
# The condition_lambda operates on the output of lcel_router_chain
lcel_routing_branch = RunnableBranch(
    (lambda x: x["destination"] == "pdf_content_qa", pdf_branch_chain),
    (lambda x: x["destination"] == "chat_history_qa", history_branch_chain),
    (lambda x: x["destination"] == "document_summary_qa", summary_branch_chain),
    (lambda x: x["destination"] == "get_last_user_question", direct_last_q_branch_chain),
    default_branch_chain_for_routing,  # Default case
)

# Combine the router chain with the branch
# The output of lcel_router_chain is fed into lcel_routing_branch
final_chain = lcel_router_chain | lcel_routing_branch

# --- Chat history list (for saving to JSON and UI display) ---
chat_history = [] # This is the Python list for raw history
if os.path.exists(CHAT_HISTORY_FILE):
    try:
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            chat_history = json.load(f)
    except Exception as e:
        st.warning(f"Error loading chat_history list from {CHAT_HISTORY_FILE}: {e}")
        chat_history = []

# Streamlit UI
st.title("Gemini PDF Chatbot with RouterChain")

if "messages" not in st.session_state:
    st.session_state.messages = []
    if chat_history:
        for item in chat_history:
            sender = item.get("sender", "Unknown")
            message_content = item.get("text") if sender == "Gemini" else item.get("question", item.get("text", ""))
            if message_content:
                 st.session_state.messages.append((sender, message_content))

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Your question:", key="user_query_input")
    submitted = st.form_submit_button("Ask")

    if submitted and user_input:
        st.session_state.messages.append(("You", user_input))
        with st.spinner("Thinking and routing your question..."):
            try:
                # The MultiRouteChain expects the raw input, the router_chain handles formatting
                chain_input_for_lcel = {"input": user_input} # lcel_router_chain expects "input"
                full_result = final_chain.invoke(chain_input_for_lcel)

                # Extract answer based on expected output keys from RetrievalQA
                answer_text = full_result.get("result")
                if answer_text is None: # Fallback for other chain types or direct LLM calls
                    answer_text = full_result.get("text")
                if answer_text is None: # Broader fallback
                    answer_text = str(full_result) # If neither 'result' nor 'text' is found

                st.session_state.messages.append(("Gemini", answer_text))

                # Update the Python list for chat history (for JSON saving)
                chat_history.append({"sender": "You", "question": user_input,})
                chat_history.append({"sender": "Gemini", "question": user_input, "text": answer_text})

                with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
                    json.dump(chat_history, f, indent=2)

                # Dynamically update the history_vectordb in session_state
                if st.session_state.get("history_vectordb"):
                    history_doc_content = f"Previous Question: {user_input}\nPrevious Answer: {answer_text}"
                    new_history_document_for_vdb = Document(page_content=history_doc_content)
                    try:
                        st.session_state.history_vectordb.add_documents([new_history_document_for_vdb])
                    except Exception as e_add_doc:
                        st.warning(f"Could not dynamically update history_vectordb: {e_add_doc}")

            except Exception as e:
                st.error(f"Error processing your question: {e}")
                error_message = f"Sorry, I encountered an error trying to answer your question.\n\n**Details:** {str(e)}"
                st.session_state.messages.append(("Gemini", error_message))
                chat_history.append({"sender": "You", "question": user_input, })
                chat_history.append({"sender": "Gemini", "question": user_input, "text": f"Error: {str(e)}"})
                with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
                    json.dump(chat_history, f, indent=2)

# Display history
st.markdown("---")
st.subheader("Chat History")
for sender, msg_text in st.session_state.messages:
    if sender == "You":
        st.markdown(f"<div style='text-align:right;background-color:#DCF8C6;padding:10px;border-radius:7px;margin-bottom:5px;'><b>You:</b> {msg_text}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align:left;background-color:#F0F0F0;padding:10px;border-radius:7px;margin-bottom:5px;'><b>Gemini:</b> {msg_text}</div>", unsafe_allow_html=True)

if st.button("Clear Chat Session and History File"):
    st.session_state.messages = []
    chat_history = []
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            os.remove(CHAT_HISTORY_FILE)
            st.success(f"Cleared {CHAT_HISTORY_FILE}")
        except Exception as e:
            st.error(f"Could not clear {CHAT_HISTORY_FILE}: {e}")
    st.rerun()
