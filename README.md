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

   Place PDF files in the `Riley Register Bulletins/Subset/` directory.

4. **Start your Ollama server** and ensure the required model (e.g., `llama3.2:latest`) is available.
5. **Run the main script:**

   ```sh
   python SimpleRAGTest.py
   ```

## Project Structure

- `SimpleRAGTest.py`: Main script for loading PDFs, building the vector store, querying, and evaluating answers.
- `requirements.txt`: Python dependencies.
- `Riley Register Bulletins/Subset/`: Directory containing PDF source documents.
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
- The script will create a FAISS vector store in the `faiss_index` directory if it does not exist.
- You can change the query in `SimpleRAGTest.py` to test different questions.
- The script prints out the retrieved documents and the generated answer, along with a similarity score.

## License

MIT
