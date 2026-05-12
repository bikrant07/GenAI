
from dotenv import load_dotenv
load_dotenv()

import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

#  PAGE CONFIG 

st.set_page_config(
    page_title="PDF RAG Assistant",
    layout="wide"
)

# CUSTOM CSS 

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.stTextInput input {
    border-radius: 10px;
    padding: 12px;
}

.response-box {
    background-color: #FFFFFF;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #333;
    margin-top: 15px;
}

.source-box {
    background-color: #161B22;
    padding: 15px;
    border-radius: 10px;
    margin-top: 10px;
    border-left: 4px solid #4CAF50;
}

</style>
""", unsafe_allow_html=True)

# TITLE 

st.title("PDF RAG Assistant")
st.markdown("Upload a PDF and ask questions from it.")

#  PDF UPLOAD 

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

#  PROCESS PDF 

if uploaded_file is not None:

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_pdf_path = tmp_file.name

    # Avoid rebuilding DB repeatedly
    if "vectorstore" not in st.session_state:

        with st.spinner("Processing PDF and creating embeddings..."):

            # Load PDF
            loader = PyPDFLoader(temp_pdf_path)
            docs = loader.load()

            # Split text
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)

            # Embeddings
            embeddings = OpenAIEmbeddings()

            # Vector DB
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings
            )

            st.session_state.vectorstore = vectorstore

        st.success("PDF processed successfully!")

    # RETRIEVER 

    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    # PROMPT

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful assistant.

            Answer ONLY from the provided context.

            If the answer is not found in the context,
            say:
            "I don't know based on the uploaded PDF."

            Context:
            {context}
            """
        ),
        ("human", "Question: {query}")
    ])

    #  LLM 

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2
    )

    #  QUERY INPUT 

    query = st.text_input(
        "Ask a question from the PDF:",
        placeholder="Example: Summarize chapter 2..."
    )

    #  ASK BUTTON 

    if st.button("Ask Question"):

        if not query.strip():
            st.warning("Please enter a question.")
        else:

            with st.spinner("Searching relevant information..."):

                # Retrieve docs
                docs = retriever.invoke(query)

                # Build context
                context = "\n\n".join(
                    [doc.page_content for doc in docs]
                )

                # Prompt
                response = prompt.format_messages(
                    query=query,
                    context=context
                )

                # LLM response
                answer = llm.invoke(response)

            #  ANSWER 

            st.subheader(" Answer")

            st.markdown(
                f"""
                <div class="response-box">
                {answer.content}
                </div>
                """,
                unsafe_allow_html=True
            )

            # ---------------- SOURCES ----------------

            st.subheader("📄 Retrieved Chunks")

            for i, doc in enumerate(docs, start=1):

                page = doc.metadata.get("page", "Unknown")

                st.markdown(
                    f"""
                    <div class="source-box">
                    <b>Chunk {i}</b> | <b>Page:</b> {page}<br><br>
                    {doc.page_content[:800]}...
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# EMPTY STATE 

else:
    st.info("Please upload a PDF to begin.")