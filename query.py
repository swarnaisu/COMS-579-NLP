import argparse
import os
from dotenv import load_dotenv, find_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI
from pinecone import Pinecone
import openai
os.environ["TOKENIZERS_PARALLELISM"] = "false"

openai.api_key = "Your-OpenAI-API-Key"

# initializing Pinecone
pc = Pinecone(api_key="Your-Pinecone-API-Key")
index = pc.Index("llama-integration")

def question_to_embedding(question):
    from llama_index.embeddings.openai import OpenAIEmbedding
    model_embedding = OpenAIEmbedding()
    vector = model_embedding.get_text_embedding(question)
    return vector


def get_answer(question):
    query_vector = question_to_embedding(question)
    # including metadata in the results
    query_result = index.query(vector=query_vector, top_k=5, include_metadata=True)

    _nodes = []
    print("Query: ", question)
    print("Retrieval")
    for i, _t in enumerate(query_result['matches']):
        try:
            _node = TextNode(text=_t['metadata']['text'])
            _nodes.append(_node)
        except Exception as e:
            print(e)

    # creating vector store index
    _index = VectorStoreIndex(_nodes)
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    
    # re-ranking
    query_engine = _index.as_query_engine(similarity_top_k=5, llm=llm)
    response = query_engine.query(question)
    print("Answer: ")
    print(str(response))


def answer_generation(texts):
    context_combining = ' '.join(texts)  # to combine texts into one continuous block

    # ensuring API key is securely configured and loaded
    load_dotenv(find_dotenv())
    try:
        # sending a completion request to OpenAI API
        response = openai.Completion.create(
            engine="gpt-3.5-turbo",  # use the latest available model
            prompt=context_combining,
            max_tokens=150,
            temperature=0.7,
            top_p=0.9,
            n=1,
            stop=None  # specify any stopping criteria (if required)
        )
        
        # extracting the first (and only) response
        answer = response.choices[0].text.strip()
        return answer
    except Exception as e:
        return f"An error occurred: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Query and retrieve answers from Pinecone index.")
    parser.add_argument("--question", type=str, required=False, help="Question to query the indexed data.")
    args = parser.parse_args()
    get_answer(args.question)

if __name__ == "__main__":
    main()

