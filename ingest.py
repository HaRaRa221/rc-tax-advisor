# ingest.py
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import ServerlessSpec
from dotenv import load_dotenv
from langchain.schema import Document
import fitz  
import os
from pinecone import Pinecone

load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "docs-rag-chatbot"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_text(text, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def index_pdf_files(pdf_paths):
    for pdf_path in pdf_paths:
        print(f"Indexing PDF: {pdf_path}...")
        text = extract_text_from_pdf(pdf_path)
        chunks = split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]
        PineconeVectorStore.from_documents(
            documents=documents,
            index_name=index_name,
            embedding=embeddings,
            namespace=os.path.basename(pdf_path)  
        )
    print("Indexing completed.")

# List of PDF files to index
pdf_files = ["knowledge/f990-1.pdf", "knowledge/i990-1.pdf"]
index_pdf_files(pdf_files)
