import argparse
import fitz
import re
import os
from pinecone import ServerlessSpec
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.embeddings.openai import OpenAIEmbedding
import pinecone

def text_embedding(chunk):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    model_embedding = OpenAIEmbedding(api_key=openai_api_key)
    vector = model_embedding.get_text_embedding(chunk)

    return vector

def pdf_reading(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()

    return text

def text_cleaning(text):
    cleaned_text = re.sub(r'\bPage \d+\b', '', text)  
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    return cleaned_text

def text_chunking(text, chunk_size=500, overlap=0.25):
    words = text.split()
    chunk_step = int(chunk_size * (1 - overlap))  # Calculate step size based on overlap
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_step) if i + chunk_size <= len(words)]

    return chunks

def upload_index(pdf_path, index):
    text = pdf_reading(pdf_path)
    cleaned_text = text_cleaning(text)
    chunks = text_chunking(cleaned_text)
    for i, chunk in enumerate(chunks):
        embedding = text_embedding(chunk)
        index.upsert(vectors=[(str(i), embedding, {'text': chunk})])  # Store embedding and text in Pinecone

def query_index(query):
    query_embedding = text_embedding(query)
    results = index.query(query_embedding, top_k=5, include_metadata=True)

    return [(match['metadata']['text'], match['score']) for match in results['matches']]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload and index a PDF file.")
    parser.add_argument("--pdf_file", type=str, required=True, help="Path to the PDF file to be uploaded and indexed.")
    
    args = parser.parse_args()

    load_dotenv()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "llama-integration"

    try:
        pc.create_index(
            index_name,
            dimension=1536,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    except pinecone.core.client.exceptions.PineconeApiException as e:
        if e.status == 409:
            print("Index already exists. Consider deleting the existing index or using a different name.")
        else:
            raise

    index = pc.Index(index_name)

    upload_index(args.pdf_file, index)
    print(f"Successfully processed and indexed {args.pdf_file}.")
