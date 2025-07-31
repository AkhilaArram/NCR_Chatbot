import os
import re
import streamlit as st
import ast
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit.components.v1 as components

# Load environment variable
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("GOOGLE_API_KEY not found. Please set it in your environment variables or a .env file.")
    st.stop()

# LangChain-compatible Gemini model (1.5 Flash)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key,
    temperature=0.2,
)

# Caching LLM calls
@st.cache_data(show_spinner=False)
def validate_with_llm(language, code_to_validate):
    validation_prompt_template = """
    You are a precise code syntax checker. Your task is to determine if the following code snippet is syntactically valid for the specified language.
    Do not execute the code or comment on its logic.

    Language: {language}
    Code Snippet:
    ```
    {code}
    ```

    Is the code syntactically valid? Respond with ONLY the word "valid" if it is correct, or provide the specific syntax error message if it is not.
    """
    validation_prompt = ChatPromptTemplate.from_template(validation_prompt_template)
    validation_chain = validation_prompt | llm
    response = validation_chain.invoke({"language": language, "code": code_to_validate})
    return response.content.strip()

@st.cache_data(show_spinner=False)
def fix_with_llm(language, code, error):
    fixer_prompt_template = """
You are an expert programmer. You have been given a code snippet with a syntax error. Your task is to fix it.
Provide only the corrected, complete, and runnable code block. Do not add any explanations, apologies, or surrounding text.

Language: {language}
Original Faulty Code:
```{language_lower}
{faulty_code}
```
The error message was:
```
{error_message}
```
Corrected Code:
"""
    fixer_prompt = ChatPromptTemplate.from_template(fixer_prompt_template)
    fixer_chain = fixer_prompt | llm
    response = fixer_chain.invoke({
        "language": language,
        "language_lower": language.lower(),
        "faulty_code": code,
        "error_message": error
    })
    return response.content.strip()

# --- Helper Functions ---
def get_language_suffix(language: str) -> str:
    suffixes = {
        "Python": ".py",
        "JavaScript": ".js",
        "Java": ".java",
        "C++": ".cpp",
    }
    return suffixes.get(language, ".txt")

def validate_code(code_to_validate: str, language: str) -> tuple[bool, str]:
    if language == "Python":
        try:
            ast.parse(code_to_validate)
            return True, "Valid"
        except SyntaxError as e:
            return False, str(e)

    result = validate_with_llm(language, code_to_validate)
    return (True, "Valid") if result.lower() == "valid" else (False, result)

def fix_code(faulty_code: str, language: str, error_message: str) -> str:
    fixed = fix_with_llm(language, faulty_code, error_message)
    match = re.search(r"```[a-zA-Z]*\s*([\s\S]+?)\s*```", fixed, re.DOTALL)
    return match.group(1).strip() if match else fixed.strip()

# Define prompt template
prompt_template = """
You are an expert programmer and a helpful assistant specializing in code analysis, translation, and visualization.

Your main task is to perform the following action on the provided code snippet: **{action}**.
The source language is: **{source_lang}**.
The target language (if applicable for the action) is: **{target_lang}**.

**Instructions:**
- **If converting or refactoring code:** The resulting code in **{target_lang}** MUST be functionally equivalent to the original.
- **If explaining code:** Provide a clear, step-by-step explanation.
- **If adding comments:** Add meaningful comments without altering the code.
- **If generating a diagram:**
    - The code must be valid Mermaid syntax.
    - Use `graph TD`, `flowchart TD`, or `sequenceDiagram`.
    - Use `[Node]` or `(Node)` instead of `{{}}`.

**Source Code ({source_lang}):**
```
{code_snippet}
```

**Result:**
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
chain = prompt | llm

# --- Streamlit UI (restored) ---
st.set_page_config(page_title="Code Converter & Visualizer", layout="wide")
st.title("👨‍💻 Code Converter & Visualizer")
st.caption("Powered by LangChain and Google Gemini")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Code")
    source_code = st.text_area("Enter your code snippet here:", height=300, key="source_code")

    languages = ["Python", "JavaScript", "Java", "C++", "Go", "SQL", "HTML", "CSS", "Auto-Detect"]
    source_lang = st.selectbox("Source Language:", languages, index=0)

    actions = {
        "Convert to Python": ("Python", "Convert"),
        "Convert to JavaScript": ("JavaScript", "Convert"),
        "Convert to Java": ("Java", "Convert"),
        "Convert to C++": ("C++", "Convert"),
        "Explain this code": ("N/A", "Explain"),
        "Generate Flowchart": ("N/A", "Generate a Mermaid flowchart"),
        "Generate Sequence Diagram": ("N/A", "Generate a Mermaid sequence diagram"),
        "Generate Graph Diagram": ("N/A", "Generate a Mermaid graph diagram"),
        "Add Comments": ("source", "Add comments to the code"),
        "Refactor / Optimize": ("source", "Refactor and optimize the code")
    }
    selected_action_key = st.selectbox("Action:", list(actions.keys()), index=4)
    auto_fix = st.toggle("Auto-fix code errors", value=True)

with col2:
    st.subheader("Output")
    if "output" not in st.session_state:
        st.session_state.output = "Output will appear here..."
    if "is_diagram" not in st.session_state:
        st.session_state.is_diagram = False

    if st.button("🚀 Process Code", use_container_width=True):
        if source_code:
            with st.spinner(f"Validating {source_lang} input code..."):
                is_valid, error_msg = validate_code(source_code, source_lang)

            if not is_valid:
                st.warning(f"Input code has a syntax error: {error_msg}")
                if auto_fix:
                    with st.spinner("Attempting to auto-fix input code..."):
                        fixed_code = fix_code(source_code, source_lang, error_msg)
                        is_valid_after_fix, new_error_msg = validate_code(fixed_code, source_lang)
                        if is_valid_after_fix:
                            st.success("Input code auto-fixed successfully!")
                            source_code = fixed_code
                            st.info("The input code was corrected before processing:")
                            st.code(source_code, language=source_lang.lower() if source_lang != "Auto-Detect" else "text")
                        else:
                            st.error(f"Auto-fix failed. Error: {new_error_msg}")
                            st.session_state.output = f"**Auto-fix failed.**\n\n**Original Error:**\n```\n{error_msg}\n```\n\n**Error after fix attempt:**\n```\n{new_error_msg}\n```"
                            st.session_state.is_diagram = False
                            st.rerun()
                else:
                    st.session_state.output = f"**Invalid {source_lang} code detected.**\n\nPlease fix the syntax error or enable 'Auto-fix'.\n\n**Error:**\n```\n{error_msg}\n```"
                    st.session_state.is_diagram = False
                    st.rerun()

            with st.spinner("Generating response..."):
                try:
                    target_lang_from_action, action_verb = actions[selected_action_key]
                    final_target_lang = source_lang if target_lang_from_action in ["source", "N/A"] else target_lang_from_action

                    initial_response = chain.invoke({
                        "action": action_verb,
                        "source_lang": source_lang,
                        "target_lang": final_target_lang,
                        "code_snippet": source_code
                    })
                    generated_content = initial_response.content

                    is_code_generation_action = "Convert" in selected_action_key or "Refactor" in selected_action_key or "Add Comments" in selected_action_key

                    if is_code_generation_action:
                        with st.spinner(f"Validating generated {final_target_lang} code..."):
                            lang_lower = final_target_lang.lower()
                            match = re.search(rf"```{lang_lower}\s*([\s\S]+?)\s*```", generated_content, re.DOTALL | re.IGNORECASE)
                            code_to_validate = match.group(1).strip() if match else generated_content.strip()
                            is_output_valid, output_error_msg = validate_code(code_to_validate, final_target_lang)

                        if not is_output_valid and auto_fix:
                            with st.spinner("Generated code has an error. Attempting to auto-fix..."):
                                fixed_output_code = fix_code(code_to_validate, final_target_lang, output_error_msg)
                                is_fixed_output_valid, _ = validate_code(fixed_output_code, final_target_lang)
                                if is_fixed_output_valid:
                                    st.success("Generated code was successfully auto-corrected.")
                                    st.session_state.output = f"```{lang_lower}\n{fixed_output_code}\n```"
                                else:
                                    st.warning("Auto-fix for generated code failed. Displaying original output.")
                                    st.session_state.output = generated_content
                        else:
                            st.session_state.output = generated_content
                    else:
                        st.session_state.output = generated_content

                    st.session_state.is_diagram = "Diagram" in selected_action_key or "Flowchart" in selected_action_key or "Graph" in selected_action_key

                except Exception as e:
                    st.session_state.output = f"An error occurred: {e}"
                    st.session_state.is_diagram = False
        else:
            st.warning("Please enter a code snippet to process.")

    if st.session_state.get("is_diagram", False):
        raw_output = st.session_state.output
        match = re.search(r"```mermaid(.*?)```", raw_output, re.DOTALL | re.IGNORECASE)
        mermaid_code = match.group(1).strip() if match else raw_output.strip()
        mermaid_code = re.sub(r';\s*$', '', mermaid_code, flags=re.MULTILINE).strip()
        mermaid_code = re.sub(r'\{(.*?)\}', r'([\1])', mermaid_code)

        st.subheader("🪵 Mermaid Raw Code (Debugging)")
        st.code(mermaid_code, language="markdown")

        if not re.match(r"^(graph\s+(TD|LR)|flowchart\s+TD|sequenceDiagram)", mermaid_code, re.IGNORECASE):
            st.warning("⚠️ Mermaid syntax not detected or invalid — using fallback diagram.")
            mermaid_code = """
graph TD
    A[Start] --> B[Decision]
    B -->|Yes| C[Option 1]
    B -->|No| D[Option 2]
    C --> E[End]
    D --> E
""".strip()

        components.html(f"""
            <div class=\"mermaid\">
            {mermaid_code}
            </div>
            <script type=\"module\">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({{ startOnLoad: true }});
            </script>
        """, height=500, scrolling=True)

        st.download_button(
            label="⬇️ Download Full Output",
            data=st.session_state.output,
            file_name="llm_output.txt",
            mime="text/plain",
            use_container_width=True,
            key="download_diagram_output"
        )

    else:
        st.markdown(st.session_state.output)
        # Add a download button for non-diagram outputs as well
        if st.session_state.output and st.session_state.output != "Output will appear here...":
            st.download_button(
                label="⬇️ Download Full Output",
                data=st.session_state.output,
                file_name="llm_output.txt",
                mime="text/plain",
                use_container_width=True,
                key="download_text_output"
            )

if st.button("Clear Output"):
    st.session_state.output = "Output will appear here..."
    st.session_state.is_diagram = False
    st.rerun() 
