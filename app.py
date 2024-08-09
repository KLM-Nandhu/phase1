import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader, Docx2TxtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import pinecone
import os
from dotenv import load_dotenv
import tempfile
import uuid

# Load environment variables
load_dotenv()

# Set up OpenAI and Pinecone
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
doc_index_name = "gradientcyber"
conv_index_name = "conversation_history"

# Initialize OpenAI embedding and LLM
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = OpenAI(temperature=0, model_name="gpt-4", openai_api_key=OPENAI_API_KEY)

# Initialize Pinecone vector stores
doc_vectorstore = Pinecone.from_existing_index(doc_index_name, embeddings)
conv_vectorstore = Pinecone.from_existing_index(conv_index_name, embeddings)

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize ConversationalRetrievalChain
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=doc_vectorstore.as_retriever(),
    memory=memory
)

# Function to process and upsert documents
def process_and_upsert_document(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.name[file.name.rfind('.'):]) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name

    if file.name.endswith('.pdf'):
        loader = PyPDFLoader(tmp_file_path)
    elif file.name.endswith('.docx'):
        loader = Docx2TxtLoader(tmp_file_path)
    else:
        st.error("Unsupported file type. Please upload PDF or DOCX files.")
        return

    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    doc_vectorstore.add_documents(texts)

    os.unlink(tmp_file_path)

    return len(texts)

# Function to store conversation in Pinecone
def store_conversation(question, answer):
    conversation_id = str(uuid.uuid4())
    conv_vectorstore.add_texts(
        texts=[f"Q: {question}\nA: {answer}"],
        metadatas=[{"conversation_id": conversation_id}],
        ids=[conversation_id]
    )

# Function to retrieve conversation history
def get_conversation_history():
    results = conv_vectorstore.similarity_search("", k=100)  # Retrieve up to 100 recent conversations
    return [doc.page_content for doc in results]

# Streamlit UI
st.title("Gradient Cyber Q&A System")

# Sidebar for file upload and buttons
with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader("Choose PDF or DOCX files", accept_multiple_files=True, type=['pdf', 'docx'])
    
    if uploaded_files:
        for file in uploaded_files:
            num_chunks = process_and_upsert_document(file)
            st.success(f"Processed and upserted {num_chunks} chunks from {file.name}")
    
    st.header("Controls")
    if st.button("Conversation History"):
        history = get_conversation_history()
        summary = llm(f"Summarize the following conversation history:\n\n{''.join(history)}")
        st.text_area("Conversation Summary", summary, height=300)
    
    if st.button("Clear History"):
        memory.clear()
        # Clear the conversation history index
        pinecone.Index(conv_index_name).delete(deleteAll=True)
        st.success("Conversation history cleared")
    
    if st.button("Reload"):
        st.experimental_rerun()

# Main area for Q&A
question = st.text_input("Ask a question about the uploaded documents:")

if question:
    result = qa({"question": question})
    answer = result['answer']
    st.write("Answer:", answer)
    
    # Store the conversation in Pinecone
    store_conversation(question, answer)

# Display conversation history
st.header("Recent Conversations")
history = get_conversation_history()
for conversation in history[:5]:  # Display the 5 most recent conversations
    st.text(conversation)
    st.markdown("---")
