import streamlit as st
import openai
import os
from dotenv import load_dotenv
import uuid
import tempfile
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import asyncio
from pinecone import Pinecone, ServerlessSpec
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(page_title="Gradient Cyber Bot", page_icon="🤖", layout="wide")

# Initialize Pinecone
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "gradientcyber"

    # Check if the index exists
    existing_indexes = pc.list_indexes().names()
    
    if index_name not in existing_indexes:
        logger.info(f"Index '{index_name}' does not exist. Attempting to create it.")
        try:
            pc.create_index(
                name=index_name,
                dimension=1536,  # OpenAI's ada-002 model uses 1536 dimensions
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-west-1'  # Adjust this to your preferred region
                )
            )
            logger.info(f"Successfully created index '{index_name}'.")
        except Exception as e:
            logger.error(f"Failed to create index '{index_name}'. Error: {str(e)}")
            st.error(f"Failed to create Pinecone index. Please check your configuration and try again. Error: {str(e)}")
            st.stop()
    else:
        logger.info(f"Index '{index_name}' already exists.")

    index = pc.Index(index_name)
    logger.info(f"Successfully connected to index '{index_name}'.")
except Exception as e:
    logger.error(f"Failed to initialize Pinecone. Error: {str(e)}")
    st.error(f"Failed to initialize Pinecone. Please check your API key and try again. Error: {str(e)}")
    st.stop()

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit app title
st.title("🤖 Gradient Cyber Bot")

# Sidebar
st.sidebar.title("Settings")

# Reset button
if st.sidebar.button("Reset Conversation"):
    st.session_state.conversation_history = []
    st.session_state.messages = []

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload a document", type=["txt", "pdf"])

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to get embeddings
def get_embedding(text):
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response['data'][0]['embedding']
    except Exception as e:
        logger.error(f"Error in getting embedding: {str(e)}")
        raise

# Function to upsert documents to Pinecone
def upsert_to_pinecone(text, metadata=None):
    try:
        embedding = get_embedding(text)
        unique_id = str(uuid.uuid4())
        index.upsert(vectors=[(unique_id, embedding, metadata)])
    except Exception as e:
        logger.error(f"Error in upserting to Pinecone: {str(e)}")
        raise

# Function to process uploaded file
def process_uploaded_file(file):
    try:
        if file.type == "text/plain":
            content = file.getvalue().decode("utf-8")
        elif file.type == "application/pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file.getvalue())
                temp_file_path = temp_file.name

            with open(temp_file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                content = "\n".join([page.extract_text() for page in pdf_reader.pages])

            os.unlink(temp_file_path)

        chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
        for chunk in chunks:
            upsert_to_pinecone(chunk, {"source": file.name})

        st.sidebar.success(f"File '{file.name}' processed and uploaded to Pinecone.")
    except Exception as e:
        logger.error(f"Error in processing uploaded file: {str(e)}")
        st.sidebar.error(f"Error processing file: {str(e)}")

# Process uploaded file
if uploaded_file:
    process_uploaded_file(uploaded_file)

# Function to expand query
def expand_query(query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates related search queries."},
                {"role": "user", "content": f"Generate 3 related search queries for: {query}"}
            ],
            max_tokens=100
        )
        expanded_queries = response.choices[0].message['content'].split('\n')
        return [query] + expanded_queries
    except Exception as e:
        logger.error(f"Error in expanding query: {str(e)}")
        return [query]  # Return original query if expansion fails

# Function to query Pinecone and get relevant context
async def get_context(query, k=3):
    try:
        query_embedding = get_embedding(query)
        results = index.query(vector=query_embedding, top_k=k, include_metadata=True)
        context = " ".join([match['metadata'].get("text", "") for match in results['matches']])
        return context, results['matches']
    except Exception as e:
        logger.error(f"Error in getting context: {str(e)}")
        return "", []

# Function to re-rank results
def re_rank_results(query, results):
    try:
        texts = [result['metadata'].get("text", "") for result in results]
        tfidf = TfidfVectorizer().fit_transform([query] + texts)
        cosine_similarities = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
        ranked_indices = np.argsort(cosine_similarities)[::-1]
        re_ranked_results = [results[i] for i in ranked_indices]
        return re_ranked_results
    except Exception as e:
        logger.error(f"Error in re-ranking results: {str(e)}")
        return results

# Function for contextual compression
def compress_context(context, query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes information."},
                {"role": "user", "content": f"Summarize the following context, focusing on information relevant to the query: '{query}'\n\nContext: {context}"}
            ],
            max_tokens=150
        )
        compressed_context = response.choices[0].message['content']
        return compressed_context
    except Exception as e:
        logger.error(f"Error in compressing context: {str(e)}")
        return context

# Function to generate response using OpenAI
async def generate_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        return response.choices[0].message['content']
    except Exception as e:
        logger.error(f"Error in generating response: {str(e)}")
        return "I'm sorry, but I encountered an error while generating a response. Please try again."

# Semantic cache
semantic_cache = {}

# Function to check semantic cache
def check_semantic_cache(query):
    try:
        query_embedding = get_embedding(query)
        for cached_query, (cached_embedding, cached_response) in semantic_cache.items():
            similarity = cosine_similarity([query_embedding], [cached_embedding])[0][0]
            if similarity > 0.95:  # Threshold for semantic similarity
                return cached_response
        return None  # Return None if no similar query is found in the cache
    except Exception as e:
        logger.error(f"Error in checking semantic cache: {str(e)}")
        return None  # Return None if there's an error

# Main chat interface
async def process_query(prompt):
    try:
        cached_response = check_semantic_cache(prompt)
        if cached_response:
            return f"(Cached) {cached_response}"

        expanded_queries = expand_query(prompt)
        contexts = []
        
        async def process_single_query(query):
            context, results = await get_context(query)
            re_ranked = re_rank_results(query, results)
            compressed = compress_context(" ".join([r['metadata'].get("text", "") for r in re_ranked[:2]]), query)
            return compressed

        with ThreadPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            contexts = await asyncio.gather(*[loop.run_in_executor(executor, process_single_query, query) for query in expanded_queries])

        full_context = " ".join(contexts)
        full_prompt = f"Context: {full_context}\n\nQuestion: {prompt}\n\nAnswer:"
        response = await generate_response(full_prompt)
        
        # Update semantic cache
        semantic_cache[prompt] = (get_embedding(prompt), response)
        
        return response
    except Exception as e:
        logger.error(f"Error in processing query: {str(e)}")
        return "I'm sorry, but I encountered an error while processing your query. Please try again."

if prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = asyncio.run(process_query(prompt))
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Update conversation history
    st.session_state.conversation_history.append({"prompt": prompt, "response": response})

# Display full conversation history
st.sidebar.title("Conversation History")
for i, exchange in enumerate(st.session_state.conversation_history):
    st.sidebar.subheader(f"Exchange {i+1}")
    st.sidebar.write(f"User: {exchange['prompt']}")
    st.sidebar.write(f"Assistant: {exchange['response']}")
    st.sidebar.write("---")
