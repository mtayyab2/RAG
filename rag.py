import numpy as np
import ollama
import os
import time
import json
from numpy.linalg import norm

def parse_file(filename):
    with open(filename, encoding="utf-8-sig") as f:
        paragraphs = []
        buffer = []
        for line in f.readlines():
            line = line.strip()
            if line:
                buffer.append(line)
            elif len(buffer):
                paragraphs.append((" ").join(buffer))
        if len(buffer):
            paragraphs.append((" ").join(buffer))
        return paragraphs

def save_embeddings(filename, embeddings):
    # create dir if it doesnt exist
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    # dump embeddings to json
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)

def load_embeddings(filename):
    # check if file exists
    if not os.path.exists(f"embeddings/{filename}.json"):
        return False
    # load embeddings from json
    with open (f"embeddings/{filename}.json", "r") as f:
        return json.load(f)

def get_embeddings(filename, modelname, chunks):
    # check if embeddings are already saved
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings
    # get embeddings from ollama
    embeddings = [
        ollama.embeddings(model=modelname, prompt=chunk)['embedding']
        for chunk in chunks
    ]
    # save embeddings
    save_embeddings(filename, embeddings)
    return embeddings

def find_similar(needle, haystack):
    needle_norm = norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

def main():
    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions
                        based on the snippets of text provided in context. Answer
                        only using the context provided, being as concise as possible.
                        If you are unsure, just say you don't know.
                        Context:
                        """

    filename = "book.txt"
    paragraphs = parse_file(filename)
    
    embeddings = get_embeddings(filename, 'phi3', paragraphs)

    prompt = input("What do you want?")
    prompt_embedding = ollama.embeddings(model='phi3', prompt=prompt)[
        "embedding"
        ]
    # most similar results
    most_similar_chunks = find_similar(prompt_embedding, embeddings)[:5]
    
    response = ollama.chat(
        model='phi3',
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT +"\n".join(paragraphs[item[1]] for item in most_similar_chunks)
            },
            {
                "role": "user", "content": prompt
            }
        ]
    )
    
    print(response["message"]["content"])
        

if __name__ == "__main__":
    main()
