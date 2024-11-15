import os
import sqlite3
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Function to load documents from the database
def load_docs_from_db():
    conn = sqlite3.connect('support_bot.db')
    cursor = conn.cursor()
    cursor.execute('SELECT title, content FROM support_docs')
    rows = cursor.fetchall()
    conn.close()
    
    documents = []
    for title, content in rows:
        doc = Document(page_content=content, metadata={"title": title})
        documents.append(doc)
    return documents

# Load and process documents
documents = load_docs_from_db()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Set up the LLM and QA chain
llm = OpenAI(model_name='gpt-4', temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Streamlit app
st.title("Intelligent Customer Support Chatbot")
st.write("How can we assist you today?")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    if message['role'] == 'user':
        st.write(f"**You:** {message['content']}")
    else:
        st.write(f"**Bot:** {message['content']}")

# User input
user_input = st.text_input("You:", key='input')

if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate response
    response = qa_chain.run(user_input)
    
    # Append bot response
    st.session_state.messages.append({"role": "bot", "content": response})
    
    # Refresh to show new messages
    st.experimental_rerun()
