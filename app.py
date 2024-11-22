import os
import sqlite3
import streamlit as st
from langchain.docstore.document import Document 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
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
t.set_page_config(
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

# Display chat history
for message in st.session_state.messages:
    if message['role'] == 'user':
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Bot:** {message['content']}")

# User input
user_input = st.text_input("Your question:", key='input')

if user_input:
    # Append user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate response from the QA chain
    try:
        response = qa_chain.run(user_input)
    except Exception as e:
        st.error("An error occurred while generating the response.")
        response = "I'm sorry, but I couldn't process your request at the moment."

    # Append bot response to chat history
    st.session_state.messages.append({"role": "bot", "content": response})

    # Refresh the page to display the new messages
    st.experimental_rerun()
