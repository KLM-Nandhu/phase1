import streamlit as st
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader, Docx2TxtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import LangChainTracer
from langchain.prompts import PromptTemplate
import os
import tempfile

# Set page config for a wider layout
st.set_page_config(layout="wide", page_title="Advanced Document QA Chatbot")

# Set up your API keys and environment variables
os.environ["OPENAI_API_KEY"] = "LANGCHAIN_API_KEY "
os.environ["PINECONE_API_KEY"] = "PINECONE_API_KEY"
os.environ["PINECONE_ENV"] = "us-east-1"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "LANGCHAIN_API_KEY"

# Initialize Pinecone
pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENV"])
doc_index_name = "gradient_cyber_customer_bot"
conv_index_name = "conversationhistory"

# Initialize LangChain components
embeddings = OpenAIEmbeddings()
doc_vectorstore = Pinecone.from_existing_index(doc_index_name, embeddings)
conv_vectorstore = Pinecone.from_existing_index(conv_index_name, embeddings)
llm = ChatOpenAI(temperature=0.3, model_name="gpt-4o")

# Initialize LangSmith tracer
tracer = LangChainTracer(project_name="advanced-document-qa-chatbot")

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Custom prompt templates
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

# Function to process question and convert to embedding
def process_question(question):
    return embeddings.embed_query(question)

# Function to check relevance in Pinecone
def check_relevance(question_embedding, vectorstore):
    results = vectorstore.similarity_search_by_vector(question_embedding, k=5)
    return results

# Function to save conversation to Pinecone
def save_conversation(question, answer):
    conv_vectorstore.add_texts([f"Q: {question}\nA: {answer}"])

# Function to generate answer
def generate_answer(question, relevant_docs):
    result = conv_qa_chain({"question": question, "chat_history": []})
    return result['answer'], result['source_documents']

# Custom CSS (same as before)
st.markdown("""
<style>
.user-avatar {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    background-color: #0068c9;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: bold;
}
.bot-avatar {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    background-color: #09ab3b;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: bold;
}
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: flex-start;
}
.user-message {
    background-color: #e6f3ff;
    margin-left: 60px;
}
.bot-message {
    background-color: #e6ffe6;
    margin-right: 60px;
}
</style>
""", unsafe_allow_html=True)

def process_document(file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name

    if file.name.endswith('.pdf'):
        loader = PyPDFLoader(tmp_file_path)
    elif file.name.endswith('.docx'):
        loader = Docx2TxtLoader(tmp_file_path)
    else:
        raise ValueError("Unsupported file format")

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    os.unlink(tmp_file_path)
    return texts

def upsert_to_pinecone(texts):
    doc_vectorstore.add_documents(texts)

def display_chat_message(text, is_user=False):
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="user-avatar">U</div>
            <div style="margin-left: 20px;">{text}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <div class="bot-avatar">AI</div>
            <div style="margin-left: 20px;">{text}</div>
        </div>
        """, unsafe_allow_html=True)

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
        # Display user message
        display_chat_message(query, is_user=True)
        st.session_state.messages.append({"text": query, "is_user": True})

        # Process question
        question_embedding = process_question(query)

        # Check relevance in conversation history
        conv_results = check_relevance(question_embedding, conv_vectorstore)

        # Check relevance in document index
        doc_results = check_relevance(question_embedding, doc_vectorstore)

        # Combine results
        all_results = conv_results + doc_results

        # Generate answer
        answer, sources = generate_answer(query, all_results)

        # Display AI response
        display_chat_message(answer, is_user=False)
        st.session_state.messages.append({"text": answer, "is_user": False})

        # Save conversation to Pinecone
        save_conversation(query, answer)

        # Display sources if any
        if sources:
            st.subheader("Sources:")
            for source in sources:
                st.write(f"- {source.metadata.get('source', 'Unknown source')}")

if __name__ == "__main__":
    main()
