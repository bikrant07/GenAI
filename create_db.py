from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma 
from langchain_openai import OpenAIEmbeddings 
from dotenv import load_dotenv
load_dotenv()

docs=PyPDFLoader("TextLoader/report.pdf").load()
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

chunks=splitter.split_documents(docs)
embeddings=OpenAIEmbeddings()

vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma-db"
)
