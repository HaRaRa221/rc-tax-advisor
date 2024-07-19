from pinecone import Pinecone
from pinecone import ServerlessSpec
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# Create Pinecone index if it doesn't exist
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
