import streamlit as st
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import LangChainTracer
from langchain.prompts import PromptTemplate
import os
import tempfile
import uuid
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(layout="wide", page_title="Gradient Cyber QA Chatbot")

# Set environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Verify environment variables
if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV, LANGCHAIN_API_KEY]):
    st.error("Missing required environment variables. Please set all required API keys.")
    st.stop()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "gradient_cyber_customer_bot"

# Initialize Pinecone
try:
    logger.info("Initializing Pinecone...")
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    logger.info("Pinecone initialized successfully.")
    
    doc_index_name = "gradientcyber"
    conv_index_name = "conversationhistory"

    # Check if indices exist
    existing_indexes = pinecone.list_indexes()
    logger.info(f"Existing Pinecone indexes: {existing_indexes}")

    if doc_index_name not in existing_indexes or conv_index_name not in existing_indexes:
        logger.error(f"Required indexes do not exist in Pinecone.")
        st.error(f"Required Pinecone indexes do not exist. Please ensure both '{doc_index_name}' and '{conv_index_name}' are created.")
        st.stop()

except Exception as e:
    logger.error(f"Error initializing Pinecone: {str(e)}")
    st.error(f"Error initializing Pinecone: {str(e)}")
    st.stop()

# Initialize LangChain components
try:
    logger.info("Initializing LangChain components...")
    embeddings = OpenAIEmbeddings()
    doc_vectorstore = LangchainPinecone.from_existing_index(doc_index_name, embeddings)
    conv_vectorstore = LangchainPinecone.from_existing_index(conv_index_name, embeddings)
    llm = ChatOpenAI(temperature=0.3, model_name="gpt-4")
    logger.info("LangChain components initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing LangChain components: {str(e)}")
    st.error(f"Error initializing LangChain components: {str(e)}")
    st.stop()

# Initialize LangSmith tracer
tracer = LangChainTracer(project_name="gradient_cyber_customer_bot")

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

# Custom prompt template
qa_template = """
You are an AI assistant tasked with answering questions based on the given context.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer: """

qa_prompt = PromptTemplate(
    template=qa_template, input_variables=["context", "question"]
)

# Initialize Conversational Retrieval Chain
conv_qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=doc_vectorstore.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": qa_prompt},
    return_source_documents=True,
    callbacks=[tracer]
)

def process_document(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    os.unlink(tmp_file_path)
    return texts

def upsert_to_pinecone(texts):
    doc_vectorstore.add_documents(texts)

def save_conversation(question, answer):
    conversation_id = str(uuid.uuid4())
    conv_vectorstore.add_texts(
        texts=[f"Q: {question}\nA: {answer}"],
        metadatas=[{"conversation_id": conversation_id}],
        ids=[conversation_id]
    )

def get_conversation_history():
    results = conv_vectorstore.similarity_search("", k=100)
    return [doc.page_content for doc in results]

def display_chat_message(text, is_user=False):
    if is_user:
        st.markdown(f"""
        <div style="background-color: #e6f3ff; border-radius: 10px; padding: 10px; margin-bottom: 10px;">
            <strong>You:</strong> {text}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color: #e6ffe6; border-radius: 10px; padding: 10px; margin-bottom: 10px;">
            <strong>AI:</strong> {text}
        </div>
        """, unsafe_allow_html=True)

def main():
    st.title("Gradient Cyber QA Chatbot")

    # Sidebar
    st.sidebar.title("Document Upload")
    
    # File upload in sidebar
    uploaded_files = st.sidebar.file_uploader("Choose PDF files", accept_multiple_files=True, type=['pdf'])

    if uploaded_files:
        for file in uploaded_files:
            texts = process_document(file)
            upsert_to_pinecone(texts)
            st.sidebar.success(f"Processed and uploaded: {file.name}")

    # Buttons in sidebar
    if st.sidebar.button("Clear Conversation"):
        memory.clear()
        st.session_state.messages = []
        st.sidebar.success("Conversation cleared!")

    if st.sidebar.button("View Conversation History"):
        history = get_conversation_history()
        st.sidebar.subheader("Conversation History")
        for conversation in history:
            st.sidebar.text(conversation)

    if st.sidebar.button("Reload"):
        st.experimental_rerun()

    # Main chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(message['text'], message['is_user'])

    # Chat input
    query = st.text_input("Ask a question about the uploaded documents:")
    if query:
        display_chat_message(query, is_user=True)
        st.session_state.messages.append({"text": query, "is_user": True})

        result = conv_qa_chain({"question": query})
        answer = result['answer']
        sources = result.get('source_documents', [])

        display_chat_message(answer, is_user=False)
        st.session_state.messages.append({"text": answer, "is_user": False})

        # Save conversation to Pinecone
        save_conversation(query, answer)

        if sources:
            st.subheader("Sources:")
            for source in sources:
                st.write(f"- {source.metadata.get('source', 'Unknown source')}")

if __name__ == "__main__":
    main()
