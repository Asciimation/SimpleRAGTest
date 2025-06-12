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


def create_vector_store(documents, save_path="faiss_index"):
    embedder = HuggingFaceEmbeddings()
    print_with_time("Creating vector store...")
    vector = FAISS.from_documents(documents, embedder)
    vector.save_local(save_path)
    print_with_time(f"Vector store saved to {save_path}")
    return vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})


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


def load_vector_store(load_path="faiss_index"):
    embedder = HuggingFaceEmbeddings()
    print_with_time(f"Loading vector store from {load_path}...")
    vector = FAISS.load_local(load_path, embedder, allow_dangerous_deserialization=True)
    return vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})


def retrieve_from_vector_store(retriever, query):
    print_with_time(f"Retrieving from vector store with query: {query}")
    retrieved_docs = retriever.invoke(query)
    print_with_time(f"Retrieved {len(retrieved_docs)} documents from vector store.")
    return retrieved_docs


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


def print_with_time(message):
    print(f"[{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def main():
    directory = "C:\\Development\\AIExperiments\\SimpleRAG\\Context"
    retriever = get_or_create_retriever(directory)
    query = "What was the Riley 9 Brooklands, when was it produced and by whom?"
    print_with_time("Demonstrate retrieval...")
    retrieved_docs = retrieve_from_vector_store(retriever, query)
    for i, doc in enumerate(retrieved_docs):
        print_with_time(f"Retrieved: {i+1}:\n{doc.page_content}\n")
        print()
    print_with_time("Creating LLM chain...")
    llm = OllamaLLM(model="llama3.2:latest")
    #llm = OllamaLLM(model="deepseek-r1:7b")
    print_with_time("Creating chain with retriever and LLM...")
    chain = create_chain(retriever, llm)
    print_with_time("Invoking chain with query...")
    answer = chain.invoke(query)
    print()
    print_with_time(f"Answer: {answer}")
    print_with_time("Loading SentenceTransformer model for similarity testing...")
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
