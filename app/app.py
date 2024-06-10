import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import time

load_dotenv()

#fetching openapi key ang groq api key
OpenAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

#title for streamlit
st.title("Chat with Llama3")

#LLM model initialization
llm = ChatGroq(groq_api_key = GROQ_API_KEY, model_name = "Llama3-8b-8192")

#prompt for the chat
prompt = ChatPromptTemplate.from_template(
    """
Answer the question based on the provided context only. Please provide the most accurate response based on the 
question 
<context> 
{context} 
<context>
Questions: {input}
If you are unable to find the answer, as a good AI Chat bot just say "I am sorry, I can't find it."

"""
)

def vector_embeddings():

    if "vectors" not in st.session_state:

        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("D:\Llama+RAG+OpenAI\data") #Data Ingestion
        st.session_state.docs = st.session_state.loader.load() #Document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 200) #breaking document by chunking
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:60]) #splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings) #vector store - OpenAI embeddings




prompt1 = st.text_input("Enter question that you want to enquire from Documents")

if st.button("Documents Embedding"):
    vector_embeddings()
    st.write("Vector Store db is ready") #seperate button to create vector store

if "vectors" not in st.session_state:
    vector_embeddings() #initialize  vector embeddings before accessing them

document_chain = create_stuff_documents_chain(llm,prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain) #retrieval chain

#seperate button for QandA
if prompt1:
    start = time.process_time() 
    response = retrieval_chain.invoke({'input':prompt1})
    print("Response Time: ", time.process_time()-start)
    st.write(response['answer'])

    #streamlit expander
    with st.expander("Document similarity search"):
        #Find the relevant chunks
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("________________________________")
