# RAG Implementation
A basic RAG implementation locally using Ollama. A utf8 text file is parsed and embeddings from its text are created and saved in a json file. Then embeddings of the prompt text are created and then both embeddings are compared using normality and most similar is printed. Prompt is taken as an input through CLI.


# Reading Assistant Using OLLama for Text Chatting

## Table of Contents
1. [Setup](#setup)
2. [Functions Description](#functions_description)
3. [Contributing](#contributing)


<hr>

## Setup
This script uses numpy, ollama, os, time, and json libraries to find text similarity using OLLama's embeddings
API.

Before running the code, make sure you have these dependencies installed:
- To install `numpy`:
- ```
  pip install numpy
  ```
- [ollama](https://ollama.com/)

To install `ollama':
```
pip install ollama
```
You would also need to have LLM models downloaded in Ollama from Ollama [library](https://ollama.com/library).
To install a model in Powershell/Ubuntu:
```
ollama pull phi3
```
To run a model:
```
ollama run phi3
```
<hr>

The first time embeddings will take a lot of time, once created and saved in a json file, it will be fast in response. 
PS it also depends on your system.

This code comprises several key functions designed for working with text data, specifically focusing on
calculating text similarity through OLLama's embedditations API. These functions are integral parts of a system
that processes files, retrieves and saves text embeddings, computes similarity scores between texts, and organizes
these tasks into a coherent workflow to streamline the analysis process. Below is an improved description of each
function with added context regarding their role in overall text similarity calculation:

### 1. `File Parsing`
This utility function takes a file as input and divides its content into individual paragraphs, represented as
strings within a list. This step serves as the preliminary preparation stage for subsequent processes that require
structured text data inputs.

### 2. `save_embeddings`
The purpose of this function is to persist the extracted embeddings in a JSON format into a specified directory,
under an 'embeddings' folder with the given filename. It ensures all necessary files are saved and organized for
easy access during later stages of analysis.

### 3. `load_embeddings`
This function is responsible for reading previously stored embeddings from a JSON file at a specified location. If
no such file exists, it returns False to indicate the absence of available data. This capability allows reuse and
consistency in reusing past embeddings work without necessitating repeated API calls or processing steps.

### 4. `get_embeddings`
This function plays a dual role: If pre-computed embeddings are present, it loads them; otherwise, it retrieves
new embeddings from the OLLama API using the provided filename and model name parameters. After acquisition, it
stores these newly fetched embeddings in a JSON file for subsequent use—facilitating efficient work with large
sets of text data by minimizing redundant computations.

### 5. `find_similar` (Text Similarity Calculation)
This function is central to the similarity analysis process, receiving an 'embedding vector' representing a
specific document and another list containing embedding vectors for multiple documents ('haystack'). It calculates
cosine similarities between the given needle embedding and each haystack embedding. The result is a sorted list of
tuples where each tuple contains a similarity score (ranging from 0 to 1) alongside the index of corresponding
haystack document, allowing users to identify closely related texts efficiently within their corpus.

By combining these functionalities, this codebase effectively supports an entire workflow for text similarity
analysis using OLLama's embeddings API, enabling robust and scalable exploration of relationships between
different documents in a dataset.

## Contributing
If you would like to contribute to this project, please fork the repository and create a pull request.

For reporting issues or suggesting improvements, feel free to open an issue on GitHub.

## Give a 🌟 to help Boost my Confidence.
