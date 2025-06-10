import os
import torch
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer

# This script demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline using LangChain.
# It is to demonstrate the basics in clear steps so is not an example of a production-ready system.
# It loads PDF documents, splits them into chunks, creates a vector store, retrieves relevant documents,
# and uses a language model to generate answers based on the retrieved context.
# The script also includes a test to check the similarity of the generated answer against a standard answer.
# It uses the FAISS vector store for efficient similarity search and the OllamaLLM for language generation.

# This function checks if the vector store exists, and if not, it creates it.
def get_or_create_retriever(directory, vector_store_path="faiss_index"):
    print_with_time(f"Current path: {os.getcwd()}")
    if not os.path.exists(os.path.join(vector_store_path, "index.faiss")):
        print_with_time("Vector store does not exist, creating...")
        docs = load_pdfs_from_directory(directory)
        documents = split_documents(docs)
        retriever = create_vector_store(documents, save_path=vector_store_path)
    else:
        print_with_time("Vector store exists, loading...")
        retriever = load_vector_store(load_path=vector_store_path)
    return retriever

# The documents are turned into embeddings (numeric representations of data) to be stored in a vector store.
# The FAISS vector store is a fast and efficient way to store and retrieve vectors.
# The embeddings are created using the HuggingFaceEmbeddings class. 
# The vector store is saved to disk so we don't have to build it each time we run.
# Building the vector store takes considerable time (hours).
# We return a retriever that can be used to query the vector store later.
# The value k used in the retriever is how many results to retrieve from the store.
def create_vector_store(documents, save_path="faiss_index"):
    embedder = HuggingFaceEmbeddings()
    print_with_time("Creating vector store...")
    vector = FAISS.from_documents(documents, embedder)
    vector.save_local(save_path)
    print_with_time(f"Vector store saved to {save_path}")
    return vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# This function reads the input documents into a list of LangChain Document objects.
def load_pdfs_from_directory(directory):
    docs = []
    print_with_time("Starting to load PDFs from directory.")
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            print_with_time(f"Loading PDF: {filepath}")
            loader = PDFPlumberLoader(filepath)
            try:
                docs.extend(loader.load())
            except Exception as e:
                print_with_time(f"Error loading {filepath}: {e}")
                print_with_time("Skipping this PDF...")
    print_with_time(f"Total number of pages loaded: {len(docs)}")
    return docs

# This function splits the documents into smaller chunks using the RecursiveCharacterTextSplitter.
# The chunk size and overlap can be adjusted as needed.
# The chunks are later used to create the vector store.
# These parameters will control how much context is given to the LLM later.
def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    documents = text_splitter.split_documents(docs)
    print_with_time(f"Number of chunks created: {len(documents)}")
    for i, doc in enumerate(documents):
        print_with_time(f"CHUNK: {i+1}\n{doc.page_content}")
    print_with_time("Chunking complete.")    
    return documents

# This function loads the vector store from disk.
# It uses the same embedding model we used to initially create it
# We return a retriever that can be used to query the vector store later.
# The value k used in the retriever is how many results to retrieve from the store.
def load_vector_store(load_path="faiss_index"):
    embedder = HuggingFaceEmbeddings()
    print_with_time(f"Loading vector store from {load_path}...")
    vector = FAISS.load_local(load_path, embedder, allow_dangerous_deserialization=True)
    return vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# This function retrieves documents that are most similar to the query from the vector store.
# It uses the retriever returned from creating or loading the vector store.
def retrieve_from_vector_store(retriever, query):
    print_with_time(f"Retrieving from vector store with query: {query}")
    retrieved_docs = retriever.invoke(query)
    print_with_time(f"Retrieved {len(retrieved_docs)} documents from vector store.")
    return retrieved_docs

# This function creates a chain that combines the retriever and the language model (LLM).
# It formats the retrieved documents and the input question into a prompt for the LLM.
# This is a form of prompt augmentation.
# The LLM is then used to generate an answer based on the context provided by the retrieved documents.
def create_chain(retriever, llm):
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def print_input(input):
        print_with_time(f"Input prompt: {input}")
        return input

    template = """
    Numbered instructions are as follows. Make sure to do them all.
    1. Use the following given context to answer the given question.
    2. Only include known facts in your answer, do not guess or make up answers.
    3. Answer in a single, succinct sentence, do not elaborate or add any additional information.
    
    Context: {context}
    Question: {question}
    """
    prompt = PromptTemplate.from_template(template)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | print_input
        | llm
        | StrOutputParser()
    )
    return chain

# Simple logging function to print messages with a timestamp.
def print_with_time(message):
    print(f"[{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# Main function to run the script.
def main():
    # Location of the PDF files to built our vector store from.
    directory = "C:\\Development\\AIExperiments\\SimpleRAG\\Riley Register Bulletins\\Subset"

    # Check if the vector store already exists, if not create it.
    # The retriever is used to query the vector store.
    retriever = get_or_create_retriever(directory)

    # Query the vector store with a test question to retrieve context.
    # Note: This step is just here to demonstrate the retrieval process.
    query = "What was the Riley 9 Brooklands, when was it produced and by whom?"
    print_with_time("Demonstrate retrieval...")
    retrieved_docs = retrieve_from_vector_store(retriever, query)
    for i, doc in enumerate(retrieved_docs):
        print_with_time(f"Retrieved: {i+1}:\n{doc.page_content}\n")
        print()

    # We 'chain' together the retriever and the LLM to answer the query.
    # We may plug in different LLM models to test their behaviour.
    print_with_time("Creating LLM chain...")
    llm = OllamaLLM(model="llama3.2:latest")
    #llm = OllamaLLM(model="deepseek-r1:7b")

    # Create the chain.
    print_with_time("Creating chain with retriever and LLM...")
    chain = create_chain(retriever, llm)

    # Invoke the chain with the query to get an answer.
    print_with_time("Invoking chain with query...")
    answer = chain.invoke(query)
    print()
    print_with_time(f"Answer: {answer}")

    # Test the answer against a standard answer using cosine similarity.
    print_with_time("Loading SentenceTransformer model for similarity testing...")
    
    # We use a different AI model here to compute the similarity score.
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    standard_answer = "The Riley Nine Brooklands was a car produced by Riley Motors between 1928 and 1932."

    print_with_time("Computing cosine similarity between the given answer and the standard answer...")
    embeddings1 = model.encode([answer], convert_to_tensor=True)
    embeddings2 = model.encode([standard_answer], convert_to_tensor=True)
    cosine_similarities = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
    print_with_time(f"Similarity score: {cosine_similarities.item()}")

    try:
        similarity_score = cosine_similarities.item()
        threshold = 0.80
        assert similarity_score > threshold, "The given answer is not sufficiently similar to the gold standard answer."
        print_with_time(f"\033[92mTest passed: The given answer score of {similarity_score:.2f} is sufficiently similar to the gold standard answer.\033[0m")
    except AssertionError as e:
        print_with_time(f"\033[91mTest failed: {e}\033[0m")
        print_with_time(f"Similarity was {similarity_score:.2f} but should be > {threshold}")

if __name__ == "__main__":
    main()
 