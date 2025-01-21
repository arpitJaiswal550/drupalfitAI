import pandas as pd
# from google.cloud import bigquery
# from storage_config import PROJECT_ID, DATASET_ID, BLOG_TABLE_NAME
import os
from dotenv import load_dotenv
# from google.auth.credentials import AnonymousCredentials
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# api_key = os.getenv("API_KEY")

# client = bigquery.Client(
#     credentials=AnonymousCredentials(),
#     project="duallens",
#     _http=bigquery.Client()._http,
# )

# client._http.headers["Authorization"] = f"Bearer {api_key}"
# table_id = f"{PROJECT_ID}.{DATASET_ID}.{BLOG_TABLE_NAME}"

# query = f"SELECT * FROM `{table_id}`"
# query_job = client.query(query)
import json

# Open and read the JSON file
with open('osl_data.json', 'r') as file:
    data = json.load(file)

data_txt = str(data)

# Define the chunking strategy
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2048,  # Adjust this value based on your needs
    chunk_overlap=516  # Overlap ensures context continuity
)

# Initialize the Gemini Embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

chunks = text_splitter.split_text(data_txt)
# print(len(chunks))

# Create a persistent Chroma vectorstore
persist_directory = "chroma_db_index_drupalfit_osl"

# Initialize the Chroma vectorstore using the documents and embeddings
vectorstore = Chroma.from_texts(
    texts=chunks,  # The list of content
    embedding=embedding_model,  # The Gemini embedding model
    # metadatas=metadatas,  # Metadata dictionary
    persist_directory=persist_directory,
)

# Save the vectorstore
vectorstore.persist()

print("Chroma vector store created!!!")
