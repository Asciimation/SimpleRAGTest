# SimpleRAG: Simple Retrieval-Augmented Generation in Python

This project demonstrates a basic Retrieval-Augmented Generation (RAG) pipeline using Python, LangChain, FAISS, HuggingFace, and Ollama. It loads a set of PDF documents, builds a vector store, retrieves relevant chunks for a query, and uses a local LLM to answer questions based on the retrieved context.

## Features

- Loads and splits PDF documents into manageable chunks
- Builds and saves a FAISS vector store for fast similarity search
- Retrieves top-k relevant document chunks for a user query
- Uses a local LLM (Ollama) to generate answers based on retrieved context
- Evaluates answer quality using cosine similarity with SentenceTransformers

## Getting Started

1. **Install Python 3.8+**
2. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

3. **Add your PDF files:**

   Place PDF files in the `Context Data/` directory.

4. **Start your Ollama server** and ensure the required model (e.g., `llama3.2:latest`) is available.
5. **Run the main script:**

   ```sh
   python SimpleRAGTest.py
   ```

## Project Structure

- `SimpleRAGTest.py`: Main script for loading PDFs, building the vector store, querying, and evaluating answers.
- `requirements.txt`: Python dependencies.
- `Context Data/Subset/`: Directory containing PDF source documents.
- `faiss_index/`: Directory where the FAISS vector store is saved.

## How It Works

1. **PDF Loading:** Reads all PDFs from the specified directory.
2. **Chunking:** Splits documents into overlapping text chunks for better retrieval.
3. **Vector Store:** Embeds chunks and saves them in a FAISS vector store for fast similarity search.
4. **Retrieval:** For a given query, retrieves the top-k most relevant chunks.
5. **Answer Generation:** Passes the retrieved context and query to a local LLM (Ollama) to generate an answer.
6. **Evaluation:** Compares the generated answer to a standard answer using cosine similarity.

## Notes

- You need to provide your own data source for vectorising and update the script to match.
- This lives in the parent directory of the code in a folder called 'Context Data'.
- The script will create a FAISS vector store in the `faiss_index` directory if it does not exist.
- You can change the query in `SimpleRAGTest.py` to test different questions.
- The script prints out the retrieved documents and the generated answer, along with a similarity score.

## License

MIT

# SimpleRAGTest Inline

This project demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline using LangChain, written in a clear, inline style for easy understanding.

## What does it do?
- Loads a directory of PDF documents.
- Splits the documents into manageable text chunks.
- Converts these chunks into vector embeddings using a HuggingFace model.
- Stores the vectors in a FAISS vector database for fast similarity search.
- When you ask a question, retrieves the most relevant chunks from the database.
- Uses a Large Language Model (LLM) to generate an answer based on the retrieved context and question.
- Compares the generated answer to a gold standard answer using similarity scoring.

## How it works
1. **Document Loading**: Reads all PDF files from a specified directory.
2. **Chunking**: Splits each document into overlapping text chunks using `RecursiveCharacterTextSplitter`.
3. **Embedding**: Converts each chunk into a vector using HuggingFaceEmbeddings (AI #1).
4. **Vector Store**: Stores all vectors in a FAISS database for efficient retrieval.
5. **Retrieval**: For a given query, retrieves the top-k most similar chunks from the vector store.
6. **LLM Answering**: Passes the retrieved context and the question to an LLM (AI #2) to generate an answer.
7. **Evaluation**: Compares the generated answer to a standard answer using SentenceTransformer (AI #3) and cosine similarity.

## Key Technologies
- **LangChain**: For chaining together document loading, splitting, retrieval, and LLM calls.
- **FAISS**: For fast vector similarity search.
- **HuggingFaceEmbeddings**: For turning text into vectors.
- **OllamaLLM**: For generating answers from context.
- **SentenceTransformer**: For evaluating answer similarity.

## How to Run
1. Place your PDF files in the `Context Data` directory.
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the script:
   ```sh
   python SimpleRAGTest_inline.py
   ```

## Notes
- The first run may take a long time as it builds the vector store. Subsequent runs are much faster.
- The script is written inline for clarity, with comments explaining each step.
- You can change the query in the script to ask different questions.
- The LLM may generate different answers each time due to its probabilistic nature.

## Example Query
```
What is one special fact about the Riley Nine cylinder head?
```

## Example Output
- Retrieved context chunks from the PDFs
- Generated answer from the LLM
- Similarity score compared to a gold standard answer

---

This project is intended for educational and experimental purposes. For production use, consider modularizing the code and adding error handling, logging, and more robust evaluation.
