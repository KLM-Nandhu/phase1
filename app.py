import streamlit as st
from pinecone import Pinecone
from openai import OpenAI
import os
from dotenv import load_dotenv
from langsmith import Client
import uuid
import asyncio
import aiohttp
from typing import List, Dict
import tempfile
import tiktoken

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-ada-002"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "gradientcyber"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = "gradientcyber_customer_bot"
MAX_TOKENS = 4096
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CACHE_EXPIRATION = 3600  # 1 hour

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
langsmith_client = Client()

# Helper functions (unchanged)
# ... (include all the helper functions from the previous implementation)

# Streamlit UI
st.set_page_config(layout="wide", page_title="Gradient Cyber Bot", page_icon="🤖")

# Custom CSS for improved UI
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #f0f4f8;
        color: #1e1e1e;
    }
    .reportview-container {
        background-color: #f0f4f8;
    }
    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 6rem;
        margin: auto;
    }
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stChatMessage:hover {
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .stChatMessage.user {
        background-color: #e6f3ff;
        border-left: 5px solid #2196F3;
    }
    .stChatMessage .content p {
        margin-bottom: 0.5rem;
        line-height: 1.6;
    }
    .stTextInput {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 1rem;
        background-color: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        z-index: 1000;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    }
    .stTextInput > div {
        display: flex;
        justify-content: space-between;
        max-width: 900px;
        margin: auto;
    }
    .stTextInput input {
        flex-grow: 1;
        margin-right: 1rem;
        border-radius: 25px;
        border: 2px solid #2196F3;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    .stTextInput input:focus {
        box-shadow: 0 0 0 2px rgba(33, 150, 243, 0.3);
        outline: none;
    }
    .stButton button {
        border-radius: 25px;
        padding: 0.75rem 1.5rem;
        background-color: #2196F3;
        color: white;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #1976D2;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .answer-card {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #4CAF50;
    }
    .answer-card h3 {
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .source-list {
        margin-top: 1rem;
        padding-left: 1.5rem;
    }
    .source-list li {
        margin-bottom: 0.5rem;
        color: #546E7A;
    }
    #scroll-to-bottom {
        position: fixed;
        bottom: 100px;
        right: 30px;
        width: 50px;
        height: 50px;
        background-color: #2196F3;
        color: white;
        border: none;
        border-radius: 50%;
        font-size: 24px;
        display: none;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        z-index: 9999;
    }
    #scroll-to-bottom:hover {
        background-color: #1976D2;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit app title
st.title("🤖 Gradient Cyber Bot")

# Sidebar
with st.sidebar:
    st.title("Options")
    
    # File upload
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf"])
    
    # Reset button
    if st.button("Reset Conversation"):
        st.session_state.messages = []

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat input
prompt = st.chat_input("What is your question?")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process the query
    with st.spinner("Thinking..."):
        # Start LangSmith trace
        with langsmith_client.trace(project_name=LANGCHAIN_PROJECT, name="query_processing") as trace:
            # Generate query embedding
            query_embedding = generate_embedding(prompt)
            
            # Retrieve relevant documents
            results = hybrid_search(index, query_embedding, prompt, top_k=5)
            
            # Process and prepare context
            context = process_results(results)
            
            # Generate answer
            answer = generate_answer(prompt, context)
            
            # Format and display answer
            formatted_answer = format_answer(answer, results)
            
            # Log metrics
            trace.add_metadata({
                "query_tokens": count_tokens(prompt),
                "response_tokens": count_tokens(answer),
                "num_results": len(results)
            })

    st.session_state.messages.append({"role": "assistant", "content": formatted_answer})
    with st.chat_message("assistant"):
        st.markdown(formatted_answer)

# Handle file upload
if uploaded_file is not None:
    with st.spinner("Processing and uploading document..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Process and upload the document
        chunks = process_document(tmp_file_path)
        asyncio.run(upload_to_pinecone(chunks, index))

        os.unlink(tmp_file_path)
    st.sidebar.success("Document processed and uploaded successfully!")

# Add scroll to bottom button
st.markdown("""
<button id="scroll-to-bottom">↓</button>
<script>
    var button = document.querySelector('#scroll-to-bottom');
    button.addEventListener('click', function() {
        window.scrollTo(0, document.body.scrollHeight);
    });
    window.addEventListener('scroll', function() {
        if ((window.innerHeight + window.scrollY) >= document.body.offsetHeight) {
            button.style.display = 'none';
        } else {
            button.style.display = 'flex';
        }
    });
</script>
""", unsafe_allow_html=True)
