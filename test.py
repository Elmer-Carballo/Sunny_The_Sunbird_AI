import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set path to your PDF file
PDF_PATH = os.getenv("PDF_FILE")

def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    
    for i, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += page_text
        else:
            print(f"⚠️ Warning: Page {i} could not be extracted.")

    print("Extracted Text Preview:", text[:500])  # Print first 500 characters
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization= True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True)

    return response["output_text"]

# Streamed response emulator
def response_generator(user_question):
    response = user_input(user_question)
    for word in response.split():
        yield word + " "

st.set_page_config(page_title="SunbirdAI", page_icon=r"logo.png", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.header(":blue[SunbirdAI]Bot",divider='orange', help = "Sunny is designed by FPU team to address all of your questions about Fresno Pacific University")
st.subheader("Hello! How can SunbirdAI help you today? 🐤")
st.caption(":blue[What a] :blue[sunny day] :blue[it is] :blue[today 🌞]")

# Process the PDF file automatically
pdf_text = get_pdf_text(PDF_PATH)
text_chunks = get_text_chunks(pdf_text)
get_vector_store(text_chunks)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Type what is on your mind"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
