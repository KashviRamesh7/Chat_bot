import streamlit as st
import pandas as pd
import docx
import fitz  # PyMuPDF for PDFs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain.chains import ConversationalRetrievalChain
import os

# Set Cohere API Key
os.environ["COHERE_API_KEY"] = "Y87rkcgxs2c35LxYDtU5gBMAV2rduMdM0QD1fg3a"
st.title("Chatbot ")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# File uploader
uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, or Excel files",
    type=["pdf", "docx", "xlsx"],
    accept_multiple_files=True
)

if uploaded_files and st.session_state.qa_chain is None:
    all_text = ""
    
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name.lower()
        file_processed = False

        # PDF
        if file_name.endswith(".pdf"):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page in doc:
                text = page.get_text("text")
                if text.strip():
                    all_text += text + "\n"
                    file_processed = True

        # DOCX
        elif file_name.endswith(".docx"):
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                if para.text.strip():
                    all_text += para.text + "\n"
                    file_processed = True

        # XLSX
        elif file_name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
            if not df.empty:
                all_text += df.to_string() + "\n"
                file_processed = True

        # Show status
        if file_processed:
            st.success(f" {uploaded_file.name} uploaded and processed successfully!")
        else:
            st.error(f" {uploaded_file.name} could not be processed (no readable content).")

    if len(all_text.strip()) == 0:
        st.error("No text found in the uploaded files.")
        st.stop()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(all_text)

    if not chunks:
        st.error("No valid chunks generated.")
        st.stop()

    # Cohere embeddings
    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # Cohere Chat
    llm = ChatCohere(model="command-r-plus", temperature=0)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )

    st.session_state.qa_chain = qa_chain

st.subheader("Ask questions about your documents")

# Chat UI
if st.session_state.qa_chain:
    # Display previous chat
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    # Input box for new question
    user_query = st.chat_input("Ask a question...")
    
    if user_query:
        st.chat_message("user").markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain({
                    "question": user_query,
                    "chat_history": [
                        (c["role"], c["content"]) for c in st.session_state.chat_history if c["role"] != "assistant"
                    ]
                })
                answer = result["answer"]
                st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
