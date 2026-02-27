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
                splitter = RecursiveCharacterTextSplitter(chunk_size=15100, chunk_overlap=0)
                split_docs = splitter.split_documents(documents)
                
                if not split_docs:
                    st.error("Document is empty or could not be parsed.")
                    st.stop()
                    
                # Process all chunks instead of just the first one
                parser = StrOutputParser()
                prompt1 = PromptTemplate(
                    template="Summarize the following set of terms and conditions in a clear and consise way: {text}",
                    input_variables=["text"]
                )
                prompt2 = PromptTemplate(
                    template="Based on the following set of terms and conditions, list out the most offensive ones: {text} \n ",
                    input_variables=["text"]
                )
                
                chain1 = prompt1 | model | parser
                chain2 = prompt2 | model | parser
                
                all_summaries = []
                all_offensive_terms = []
                
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_chunks = len(split_docs)
                for i, chunk in enumerate(split_docs):
                    status_text.text(f"Processing chunk {i+1} of {total_chunks}...")
                    
                    # 1. Summarize the chunk
                    chunk_summary = chain1.invoke({"text": chunk.page_content})
                    all_summaries.append(chunk_summary)
                    
                    # 2. Find offensive terms in the chunk (user changed prompt to use {text})
                    chunk_offensive = chain2.invoke({"text": chunk.page_content})
                    if chunk_offensive and len(chunk_offensive.strip()) > 5:
                        all_offensive_terms.append(f"**From Part {i+1}:**\n{chunk_offensive}")
                        
                    progress_bar.progress((i + 1) / total_chunks)
                
                status_text.text("Finalizing results...")
                
                # Combine results
                final_summary = "\n\n".join(all_summaries)
                final_offensive = "\n\n".join(all_offensive_terms) if all_offensive_terms else "No highly offensive terms found."
                
                st.session_state.summary = final_summary
                st.session_state.offensive_terms = final_offensive
                
                # clear progress UI
                progress_bar.empty()
                status_text.empty()
                
                # Initialize display messages
                st.session_state.messages = [
                    {"role": "assistant", "content": f"**Summary of all {total_chunks} sections:**\n\n{final_summary}"},
                    {"role": "assistant", "content": f"**Offensive Terms across all sections:**\n\n{final_offensive}"}
                ]
                
                # Initialize chat history for the model
                st.session_state.chat_history = [final_summary, final_offensive]
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
