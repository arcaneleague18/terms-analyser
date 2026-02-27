import streamlit as st
import tempfile
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate 
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="T&C Analyzer", page_icon="⚖️", layout="wide")

st.title("⚖️ Terms & Conditions Analyzer")
st.write("Upload a Terms & Conditions document (TXT) to summarize and find offensive terms.")

# --- Setup LLM & Chains ---
@st.cache_resource
def get_model():
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen3-Coder-Next",
        task="text-generation",
        temperature=0,
    )
    return ChatHuggingFace(llm=llm)

try:
    model = get_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Initialize session state variables
if "summary" not in st.session_state:
    st.session_state.summary = None
if "offensive_terms" not in st.session_state:
    st.session_state.offensive_terms = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File Uploader
uploaded_file = st.file_uploader("Upload terms.txt", type=["txt"])

if uploaded_file is not None and st.session_state.summary is None:
    if st.button("Analyze Document"):
        with st.spinner("Analyzing document..."):
            # Save uploaded file temporarily to use with TextLoader
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                loader = TextLoader(tmp_path)
                documents = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                split_docs = splitter.split_documents(documents)
                
                if not split_docs:
                    st.error("Document is empty or could not be parsed.")
                    st.stop()
                    
                # The original code only takes the first chunk
                first_chunk = split_docs[0].page_content
                
                parser = StrOutputParser()
                prompt1 = PromptTemplate(
                    template="Summarize the following set of terms and conditions in a clear and consise way: {text}",
                    input_variables=["text"]
                )
                prompt2 = PromptTemplate(
                    template="Based on the following summary of terms and conditions, list out the most offensive ones: {summary} \n ",
                    input_variables=["summary"]
                )
                
                chain1 = prompt1 | model | parser
                chain2 = prompt2 | model | parser
                
                result1 = chain1.invoke({"text": first_chunk})
                result2 = chain2.invoke({"summary": result1})
                
                st.session_state.summary = result1
                st.session_state.offensive_terms = result2
                
                # Initialize display messages
                st.session_state.messages = [
                    {"role": "assistant", "content": f"**Summary:**\n{result1}"},
                    {"role": "assistant", "content": f"**Offensive Terms:**\n{result2}"}
                ]
                
                # Initialize chat history for the model
                st.session_state.chat_history = [result1, result2]
                st.rerun()
                
            finally:
                os.unlink(tmp_path) # Clean up temp file

if st.session_state.summary is not None:
    st.markdown("### Analysis Results")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📝 Summary")
        st.info(st.session_state.summary)
    with col2:
        st.subheader("🚩 Offensive Terms")
        st.warning(st.session_state.offensive_terms)
        
    st.divider()
    st.markdown("### 💬 Chat about the Document")
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question about the Terms & Conditions..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        
        # Add user message to UI and model chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chatresult = model.invoke(st.session_state.chat_history)
                response_text = chatresult.content
                st.markdown(response_text)
                
        # Add AI response to UI and model chat history
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.session_state.chat_history.append(response_text)
