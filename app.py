import os
import sqlite3
import streamlit as st
from langchain.docstore.document import Document 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Connect to SQLite
def load_faqs_from_db(db_path='faqs.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT question, answer FROM faqs')
    rows = cursor.fetchall()
    conn.close()
    documents = []
    for question, answer in rows:
        content = f"Q: {question}\nA: {answer}"
        documents.append(Document(page_content=content))
    return documents
    
# Load documents from the database
documents = load_faqs_from_db()

# Process FAQs with LangChain
#Split docs into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Create Embeddings and Build Vector Store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

#Initialize llm
llm = OpenAI(model_name='gpt-4', temperature=0)
#Set up RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # You can also use 'map_reduce' or 'refine' if needed
    retriever=vectorstore.as_retriever()
)

# Streamlit UI
st.set_page_config(
    page_title="Walmart+ Customer Support Chatbot",
    page_icon="🛍️",
    layout="wide",
)

import os
import sqlite3
import streamlit as st
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Set the OpenAI API Key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Function to load FAQs from the database
def load_faqs_from_db(db_path='faqs.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT question, answer FROM faqs')
    rows = cursor.fetchall()
    conn.close()
    documents = []
    for question, answer in rows:
        content = f"Q: {question}\nA: {answer}"
        documents.append(Document(page_content=content))
    return documents

# Load documents from the database
documents = load_faqs_from_db()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Create embeddings and build the vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Set up the RetrievalQA chain
llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0)  # Using a more cost-effective model
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Set the page configuration
st.set_page_config(
    page_title="Walmart+ Customer Support Chatbot",
    page_icon="🛍️",
    layout="wide",
)

# App title and description
st.title("🛍️ Walmart+ Customer Support Chatbot")
st.write("Ask any question about Walmart+ and get instant answers!")

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history using new chat elements
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])

# Accept user input with chat_input
if prompt := st.chat_input("Your question"):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate response
    with st.chat_message("assistant"):
        try:
            response = qa_chain.run(prompt)
        except Exception as e:
            st.error("An error occurred while generating the response.")
            response = "I'm sorry, but I couldn't process your request at the moment."
        st.write(response)
        # Append assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
