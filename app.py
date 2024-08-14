import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import os
from dotenv import load_dotenv
import uuid
import asyncio
import aiohttp
from typing import List, Dict, Any
import tempfile
import tiktoken
from concurrent.futures import ThreadPoolExecutor
import time
from functools import lru_cache
import hashlib
import json
import PyPDF2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "gradientcyber"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME, serverless=ServerlessSpec())

# Try to initialize LangSmith client, but provide a fallback if it fails
try:
    from langsmith import Client
    langchain_client = Client(api_key=LANGCHAIN_API_KEY)
except Exception as e:
    logger.warning(f"Failed to initialize LangSmith client: {e}")
    langchain_client = None

# Helper function for tracing (with fallback)
def trace_function(func):
    def wrapper(*args, **kwargs):
        if langsmith_client:
            with langsmith_client.trace(project_name="RAG_System", name=func.__name__) as trace:
                result = func(*args, **kwargs)
                trace.add_metadata({"function": func.__name__})
                return result
        else:
            return func(*args, **kwargs)
    return wrapper

@trace_function
def generate_embedding(text: str) -> List[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

@trace_function
def hybrid_search(index, query_embedding: List[float], query_text: str, top_k: int) -> List[Dict]:
    try:
        logger.info(f"Querying Pinecone with top_k={top_k}")
        logger.info(f"Query text: {query_text}")
        logger.info(f"Query embedding shape: {len(query_embedding)}")
        
        start_time = time.time()
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter={"text": {"$contains": query_text}}
        )
        end_time = time.time()
        
        logger.info(f"Query time: {end_time - start_time} seconds")
        logger.info(f"Number of results: {len(results['matches'])}")
        
        return results['matches']
    except Exception as e:
        logger.error(f"Error in hybrid_search: {str(e)}")
        st.error(f"An error occurred while searching: {str(e)}")
        return []

def process_results(results: List[Dict]) -> str:
    return " ".join([result['metadata'].get("text", "") for result in results])

@lru_cache(maxsize=1000)
def semantic_cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

@trace_function
@lru_cache(maxsize=100)
def generate_answer(prompt: str, context: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the user's question."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
        ]
    )
    return response.choices[0].message.content

def format_answer(answer: str, results: List[Dict]) -> str:
    formatted_answer = f"{answer}\n\nSources:\n"
    for i, result in enumerate(results, 1):
        formatted_answer += f"{i}. {result['metadata'].get('source', 'Unknown')}\n"
    return formatted_answer

def process_document(file_path: str) -> List[str]:
    chunks = []
    if file_path.endswith('.pdf'):
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text = page.extract_text()
                chunks.extend([text[i:i+1000] for i in range(0, len(text), 900)])
    else:
        with open(file_path, 'r') as file:
            text = file.read()
            chunks = [text[i:i+1000] for i in range(0, len(text), 900)]
    return chunks

async def upload_to_pinecone(chunks: List[str], index):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, chunk in enumerate(chunks):
            embedding = await generate_embedding_async(chunk, session)
            task = index.upsert(vectors=[(str(uuid.uuid4()), embedding, {"text": chunk, "source": f"Document chunk {i+1}"})])
            tasks.append(task)
        await asyncio.gather(*tasks)

async def generate_embedding_async(text: str, session) -> List[float]:
    async with session.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={"model": "text-embedding-ada-002", "input": text}
    ) as response:
        result = await response.json()
        return result["data"][0]["embedding"]

@trace_function
def expand_query(query: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Expand the given query to improve search results."},
            {"role": "user", "content": f"Expand this query: {query}"}
        ]
    )
    return response.choices[0].message.content

@trace_function
def rerank_results(results: List[Dict], query: str) -> List[Dict]:
    # Implement re-ranking logic here
    # This is a placeholder implementation
    return sorted(results, key=lambda x: x['score'], reverse=True)

@trace_function
def apply_contextual_compression(context: str) -> str:
    # Implement contextual compression logic here
    # This is a placeholder implementation
    return context[:1000] if len(context) > 1000 else context

# Streamlit UI
st.set_page_config(layout="wide", page_title="Gradient Cyber Bot", page_icon="🤖")

# Custom CSS
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

    # Conversation History button
    if st.button("View Conversation History"):
        if "messages" in st.session_state:
            history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
            st.text_area("Conversation History", history, height=300)
        else:
            st.warning("No conversation history available.")

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
        try:
            # Generate query embedding
            query_embedding = generate_embedding(prompt)
            
            # Optionally expand the query
            expanded_query = expand_query(prompt)
            
            # Retrieve relevant documents
            results = hybrid_search(index, query_embedding, expanded_query, top_k=5)
            
            if not results:
                st.warning("No results found. The system might be experiencing issues.")
                formatted_answer = "I'm sorry, but I couldn't retrieve any relevant information at the moment. Please try again later or rephrase your question."
            else:
                # Re-rank results
                reranked_results = rerank_results(results, prompt)
                
                # Process and prepare context
                context = process_results(reranked_results)
                compressed_context = apply_contextual_compression(context)
                
                # Generate answer
                answer = generate_answer(prompt, compressed_context)
                
                # Format and display answer
                formatted_answer = format_answer(answer, reranked_results)

        except Exception as e:
            logger.error(f"Error in query processing: {str(e)}")
            st.error(f"An error occurred while processing your query: {str(e)}")
            formatted_answer = "I'm sorry, but an error occurred while processing your request. Please try again later."

    st.session_state.messages.append({"role": "assistant", "content": formatted_answer})
    with st.chat_message("assistant"):
        st.markdown(formatted_answer)

# Handle file upload
if uploaded_file is not None:
    with st.spinner("Processing and uploading document..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf' if uploaded_file.type == 'application/pdf' else '.txt') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # Process and upload the document
            chunks = process_document(tmp_file_path)
            asyncio.run(upload_to_pinecone(chunks, index))
            st.sidebar.success("Document processed and uploaded successfully!")
        except Exception as e:
            logger.error(f"Error in document processing: {str(e)}")
            st.sidebar.error(f"An error occurred while processing the document: {str(e)}")
        finally:
            os.unlink(tmp_file_path)
