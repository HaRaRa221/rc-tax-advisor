from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

print(f"Pinecone API Key: {pinecone_api_key}")
print(f"OpenAI API Key: {openai_api_key}")

# You can now use these variables in your code, for example:
import pinecone
import openai

# Initialize Pinecone
pinecone.init(api_key=pinecone_api_key)

# Initialize OpenAI
openai.api_key = openai_api_key
