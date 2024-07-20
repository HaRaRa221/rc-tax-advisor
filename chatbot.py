# chatbot.py
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Define the Pinecone index name and namespace
index_name = "docs-rag-chatbot"
namespace = "f990-1.pdf"  # Change to your namespace

# Initialize LangChain components
knowledge = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    namespace=namespace,  
    embedding=embeddings
)

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-3.5-turbo",
    temperature=0.0
)

# Define a detailed persona for the bot
persona = (
    "You are a highly experienced tax advisor with over 20 years of experience in the field. "
    "You specialize in various areas of tax law, including individual income tax, corporate tax and nonprofit organizations "
    "international tax, and estate planning. You are well-versed in both federal and state tax regulations. "
    "Your approach is both thorough and practical, ensuring that your responses are not only accurate but also "
    "easily understandable to individuals with varying levels of tax knowledge. "
    "You stay updated with the latest changes in tax laws and regulations and are skilled in providing detailed, "
    "comprehensive advice on tax-related matters. "
    "Your goal is to assist users by providing accurate information, answering their questions in a clear and "
    "concise manner, and guiding them through common tax issues. "
    "If a question falls outside your expertise or is too complex to answer accurately ask for additional information,"
    "if the user asks about generating / filling out a 990 form ask him about all the information that you need to do so, and return a csv comma separated output"
)

# Define the custom prompt template
template = '''

{context}

Respond in the persona of %s

Question: {question}
Answer:
'''

prompt = PromptTemplate(
    template=template % persona,
    input_variables=['context', 'question']
)

# Initialize RetrievalQA Chain
chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=knowledge.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

def chatbot(query):
    response = chain({"query": query})
    return response

if __name__ == "__main__":
    while True:
        query = input("Enter your question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        response = chatbot(query)
        print(response)
