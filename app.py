import os
import json
import PyPDF2
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import openai
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI and Pinecone
openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "gradientcyber"
# If the index doesn't exist, create it
if index_name not in pc.list_indexes():
    pc.create_index(
        index_name,
        dimension=1536,  # dimensionality of text-embedding-ada-002
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-1'  # replace with your preferred region
        )
    )

# Connect to the index
index = pc.Index(index_name)

def extract_and_structure_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    structured_data = []
    
    for i, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        page_data = {
            "page_number": i + 1,
            "content": text,
            "metadata": {
                "source": pdf_file.name,
                "page": i + 1
            }
        }
        structured_data.append(page_data)
    
    return json.dumps(structured_data)

def get_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response['data'][0]['embedding']

def process_and_upload_pdf(pdf_file):
    json_data = extract_and_structure_pdf(pdf_file)
    structured_data = json.loads(json_data)
    
    vectors_to_upsert = []
    for item in structured_data:
        embedding = get_embedding(item['content'])
        metadata = {
            "page_number": item['page_number'],
            "source": item['metadata']['source'],
            "content": item['content']
        }
        vectors_to_upsert.append((f"{pdf_file.name}_page_{item['page_number']}", embedding, metadata))
    
    index.upsert(vectors=vectors_to_upsert)

    return len(structured_data)

def search_pinecone(query, top_k=3):
    query_embedding = get_embedding(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return results

def generate_response(query, context):
    prompt = f"Based on the following context, please answer the question: {query}\n\nContext:\n{context}"
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.3,
    )
    return response.choices[0].message['content'].strip()

# Streamlit UI
st.title("Customer Support Chatbot")

# Sidebar for PDF upload
st.sidebar.title("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    if st.sidebar.button("Process and Upload PDF"):
        with st.spinner("Processing and uploading PDF..."):
            pages_uploaded = process_and_upload_pdf(uploaded_file)
        st.sidebar.success(f"Successfully processed and uploaded {pages_uploaded} pages to Pinecone.")
        
        # Display the JSON structure
        json_data = extract_and_structure_pdf(uploaded_file)
        st.sidebar.json(json_data)

# Main chat interface
user_input = st.text_input("Ask a question about the customer data:")

if user_input:
    with st.spinner("Searching for relevant information..."):
        search_results = search_pinecone(user_input)
        context = "\n".join([match['metadata']['content'] for match in search_results['matches']])
    
    with st.spinner("Generating response..."):
        response = generate_response(user_input, context)
    
    st.write("Chatbot:", response)

    with st.expander("View related context"):
        st.write(context)

    # Display JSON representation of the last search results
    with st.expander("View JSON representation of search results"):
        st.json(search_results)
