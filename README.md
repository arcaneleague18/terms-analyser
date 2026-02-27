# ⚖️ Terms & Conditions Analyzer

## Overview
The **Terms & Conditions Analyzer** is an AI-powered application designed to help users quickly understand long and complex legal documents. By leveraging Large Language Models (LLMs), this tool bridges the gap between complex legal jargon and everyday users, mitigating the risks associated with blind agreements.

## Features
- **Document Summarization:** Upload a Terms & Conditions text file (`.txt`) and get a clear, concise summary within seconds.
- **Risk Extraction:** Automatically identifies and highlights the most offensive, unfair, or highly risky clauses hidden in the fine print.
- **Interactive Chat:** An integrated chatbot interface allows you to ask follow-up questions about the document and get immediate answers.

## Technologies Used
- **Frontend Framework:** Streamlit (for building an interactive and easy-to-use web app)
- **AI/LLM Orchestration:** Langchain (TextLoader, RecursiveCharacterTextSplitter, PromptTemplate, StrOutputParser)
- **Language Model:** Qwen/Qwen3-Coder-Next via HuggingFace Endpoint
- **Environment Management:** Python `dotenv` for handling API keys

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd "t&c checker"
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   Make sure you have the necessary libraries installed. 
   ```bash
   pip install streamlit langchain langchain-huggingface langchain-community python-dotenv
   ```

4. **Set up environment variables:**
   Create a `.env` file in the root directory and add your Hugging Face API Token:
   ```env
   HUGGINGFACEHUB_API_TOKEN=your_hugging_face_token_here
   ```

## Usage

1. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

2. **Interact with the App:**
   - Open the provided local URL in your web browser (usually `http://localhost:8501`).
   - Click the "Browse files" button to upload a `terms.txt` file containing the terms and conditions you want to analyze.
   - Click **Analyze Document** to generate the summary and extract offensive terms.
   - Use the chat input box at the bottom to ask custom questions about the uploaded document.

## Project Structure
- `app.py`: The main Streamlit application containing the UI and Langchain integration.
- `.env`: Environment variables configuration file (store your API keys here).

