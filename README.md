# RAG
A basic RAG implementation locally using Ollama.

# Reading Assistant Using OLLama for Text Similarity

## Table of Contents
1. [Setup](#setup)
2. [Functions Description](#functions_description)
3. [Example Usage](#example-usage)
4. [Contributing](#contributing)
5. [License](#license)

<hr>

## Setup
This script uses numpy, ollama, os, time, and json libraries to find text similarity using OLLama's embeddings
API.

Before running the code, make sure you have these dependencies installed:
- `numpy` (version 1.20 or newer)
- [ollama](https://pypi.org/project/ollama/)

To install ollama:
```
pip install ollama
```

<hr>

## Functions Description
This section describes the purpose of each function in the code and how they work together to find text similarity
using OLLama's embeddings API.

### 1. `parse_file`
Parses a given file into paragraphs, returning a list of strings where each string represents a single paragraph.

### 2. `save_embeddings`
Saves the given embeddings in JSON format inside an "embeddings" folder with the provided filename. The function
creates the folder if it doesn't exist and saves the file after reading all files from current working directory.

### 3. `load_embeddings`
Loads and returns embeddings stored in a JSON file at the specified location. If no such file is found, it returns
False.

### 4. `get_embeddings`
Loads existing embeddings if available or retrieves them from OLLama's API using provided filename and model name.
After that, saves the newly retrieved embeddings into a JSON file for future use.

### 5. `find_similar`
Takes two inputs - an embedding vector (`needle`) and another list of embedding vectors (`haystack`). It returns a
sorted list containing tuples with similarity score and corresponding index from the input list.

<hr>

## Example Usage
This example demonstrates how to use this script by loading embeddings, finding similar paragraphs, and using
OLLama's API for chat functionality.

```python
import time

filename = "book.txt"
paragraphs = parse_file(filename)
embeddings = get_embeddings(filename, 'phi3', paragraphs)
prompt = input("What do you want?")
prompt_embedding = ollama.embeddings(model='phi3', prompt=prompt)['embedding']
most_similar_chunks = find_similar(prompt_embedding, embeddings)[:5]

response = ollama.chat(
    model='phi3',
    messages=[
        {
            "role": "system",
            "content": "\n".join([paragraphs[item[1]] for item in most_similar_chunks])
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print(response["message"]["content"])
```
<hr>

## Contributing
If you would like to contribute to this project, please fork the repository and create a pull request.

For reporting issues or suggesting improvements, feel free to open an issue on GitHub.

<hr>

## License
This code is available under the MIT license. Please include this notice in all copies or redistributions of the
software.
```
