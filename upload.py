import argparse
import fitz
import re
import pinecone
import os

from dotenv import load_dotenv, find_dotenv
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
load_dotenv(find_dotenv())

def text_embedding(chunk):
    from llama_index.embeddings.openai import OpenAIEmbedding
    model_embedding = OpenAIEmbedding()
    vector = model_embedding.get_text_embedding(chunk)
    return vector

#Initialize pinecone
pc = Pinecone(api_key="e9594329-56b9-4403-81c7-9cd2a0dc8bd0")
index = pc.Index("llama-integration")

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


def upload_index(pdf_path):
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
    upload_index(args.pdf_file)
    print(f"Successfully processed and indexed {args.pdf_file}.")
