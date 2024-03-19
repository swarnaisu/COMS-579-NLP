import sys
import os
import argparse
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModel
import torch
from pinecone import Pinecone

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Initialize Pinecone
pc = Pinecone(api_key="8ff82485-278b-402d-b51f-fb33a8ab0aaa")
index = pc.Index("rag-system-index")

# Placeholder for the actual LlamaIndex embedding function
def get_embeddings(text_chunk):
    """
    Generates an embedding for a text chunk using Hugging Face's transformers.
    """
    # Tokenize the text chunk
    inputs = tokenizer(text_chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Move the tensors to the same device as the model
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}

    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Pool the outputs into a single mean vector
    embeddings = outputs.last_hidden_state.mean(dim=1)

    # Convert to list and return
    return embeddings.squeeze().tolist()

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text_content = []
    
    for page in reader.pages:
        text_content.append(page.extract_text())
    
    return text_content

def upload_pdf(file_path):
    # Read the PDF file
    text_chunks = read_pdf(file_path)
    print(text_chunks)

    embeddings = get_embeddings(text_chunks)
    print(embeddings)

    # Store the embeddings and their associated text chunks in Pinecone
    for i, embedding in enumerate(embeddings):
        # The upsert method adds or updates a vector
        index.upsert(vectors=[(str(i), embedding)])

    print(f"PDF '{file_path}' has been read, indexed, and stored in Pinecone.")

def load_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf_file',
                         help="PDF file to add to the folder")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    pdf_file_path = load_arguments().pdf_file
    print(f"file name: {pdf_file_path}")
    if not os.path.isfile(pdf_file_path):
        print(f"File {pdf_file_path} does not exist.")
        sys.exit(1)

    upload_pdf(pdf_file_path)
