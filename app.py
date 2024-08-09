import streamlit as st
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader, Docx2TxtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import LangChainTracer
from langchain.prompts import PromptTemplate
import os
import tempfile

# Streamlit page configuration
st.set_page_config(layout="wide", page_title="Advanced Document QA Chatbot")

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("us-east-1")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Check if environment variables are set
if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV, LANGCHAIN_API_KEY]):
    st.error("Please set all required environment variables.")
    st.stop()

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
doc_index_name = "gradient_cyber_customer_bot"
conv_index_name = "conversationhistory"

# Initialize LangChain components
embeddings = OpenAIEmbeddings()
doc_vectorstore = Pinecone.from_existing_index(doc_index_name, embeddings)
conv_vectorstore = Pinecone.from_existing_index(conv_index_name, embeddings)
llm = ChatOpenAI(temperature=0.3, model_name="gpt-4")

# Initialize LangSmith tracer
tracer = LangChainTracer(project_name="advanced-document-qa-chatbot")

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

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

# Function definitions (process_question, check_relevance, save_conversation, generate_answer)
# ... [Keep these functions as they were in the original code] ...

# Custom CSS
st.markdown("""
<style>
.user-avatar { width: 50px; height: 50px; border-radius: 50%; background-color: #0068c9; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; }
.bot-avatar { width: 50px; height: 50px; border-radius: 50%; background-color: #09ab3b; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; }
.chat-message { padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex; align-items: flex-start; }
.user-message { background-color: #e6f3ff; margin-left: 60px; }
.bot-message { background-color: #e6ffe6; margin-right: 60px; }
</style>
""", unsafe_allow_html=True)

# Main function
def main():
    st.title("Advanced Document QA Chatbot")

    # Sidebar
    st.sidebar.title("Document Upload and Controls")
    
    # File upload in sidebar
    uploaded_files = st.sidebar.file_uploader("Choose PDF or DOCX files", accept_multiple_files=True, type=['pdf', 'docx'])

    if uploaded_files:
        for file in uploaded_files:
            texts = process_document(file)
            upsert_to_pinecone(texts)
            st.sidebar.success(f"Processed and uploaded: {file.name}")

    # Buttons in sidebar
    if st.sidebar.button("Clear Conversation"):
        memory.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.sidebar.success("Conversation cleared!")

    if st.sidebar.button("Reload"):
        st.experimental_rerun()

    # Main chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(message['text'], message['is_user'])

    # Chat input
    query = st.text_input("Ask a question:")
    if query:
        # Process query and generate response
        # ... [Keep this part as it was in the original code] ...

if __name__ == "__main__":
    main()
