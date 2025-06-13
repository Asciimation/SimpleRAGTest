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

# Simple logging function to print messages with a timestamp.
def print_with_time(message):
    print(f"[{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# This script demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline using LangChain.
# It loads PDF documents, splits them into chunks, creates a vector store, retrieves relevant documents,
# and uses a language model to generate answers based on the retrieved context and a question.

# It is written inline to make it easier for non coders to read through.

# Location of the PDF files to build our vector store from.
directory = os.path.join(os.getcwd(), "Context Data")
vector_store_path = "faiss_index"
print_with_time(f"Current path: {os.getcwd()}")

# Check if the vector store exists, and if not, create it.
if not os.path.exists(os.path.join(vector_store_path, "index.faiss")):
    print_with_time("Vector store does not exist, creating...")
    docs = []
    print_with_time("Starting to load PDFs from directory.")
    
    # Read each PDF file in the directory and load its content.
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

    # Split documentsinto chunks (blocks of text) based on character sizes.
    # The chunk size and overlap can be changed and control the size of the text chunks.
    # The chunks are the pieces of context we use to answer questions.
    # Different methods could be used here: splitting into tokens, sentences, paragraphs, etc.
    chunk_size = 1000
    chunk_overlap = 200
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

    # Create vector store - a specialised database for vectors 
    # Vectors are numerical representations of the text chunks.
    # The FAISS vector store is a fast and efficient way to store and retrieve vectors.
    # The 'embedder' we use to create the chunks is itself an AI (AI number 1 in this example).
    # Creating the vector store takes considerable time so here we only create it
    # if it does not already exist.
    # If it exists we read from the esisting one.
    # We create a 'retriever' which we use to get things from the vector store.
    # When we create the retriever we specify how many chunks to retrieve: k = 3
    print_with_time("Creating embeddings for the documents...")
    embedder = HuggingFaceEmbeddings()
    print_with_time("Creating vector store...")
    vector = FAISS.from_documents(documents, embedder)
    vector.save_local(vector_store_path)
    print_with_time(f"Vector store saved to {vector_store_path}")
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
else:
    print_with_time("Vector store exists, loading...")
    embedder = HuggingFaceEmbeddings()
    vector = FAISS.load_local(vector_store_path, embedder, allow_dangerous_deserialization=True)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Query the vector store with a test question to retrieve context.
# This query is what a user might type into a chatbot.
query = "What was the Riley 9 Brooklands, when was it produced and by whom?"
#query = "What shape are the combustion chambers in a Riley Nine cylinder head?"

# We use the retriever to return k chunks based on the query.
# This is just here to show how it works. We later do this again as part of the chain.
# The retriever is deterministic, we always get the same chunks for the qiven query no matter how 
# many times we call it!   
print_with_time("Use retriever to get chunks...")
retrieved_docs = retriever.invoke(query)
for i, doc in enumerate(retrieved_docs):
    print_with_time(f"Retrieved: {i+1}:\n{doc.page_content}\n")
    print() 

# Create the LLM chain.
# An LLM takes a single input of text.
# Chaining is a way to build up what the input to the LLM will be from different smaller steps,
# and then format the poutput from the LLM.
# The LLM is an AI (AI number 2 in this example).

print_with_time("Creating LLM chain...")
llm = OllamaLLM(model="llama3.2:latest")

print_with_time("Creating chain with retriever and LLM...")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def print_input(input):
    print_with_time(f"Input prompt: {input}")
    return input

# We use a template to help build the text to feed into the LLM.
# The template lets us add extra instructions as well as context.
template = """
Numbered instructions are as follows. Make sure to do them all.
1. Use the following given context to answer the given question.
2. Only include known facts in your answer, do not guess or make up answers.
3. Answer in a single, succinct sentence. Do not elaborate or add any additional information.
4. Ensure the answer is polite.

Context: {context}
Question: {question}
"""
prompt = PromptTemplate.from_template(template)

# Build our chain. This is only setting up the chain, we actually use it later.
# The steps below:
    # Use the retriever with the passed in query to retrieve chunks.
    # Built the prompt to be fed into the LLM from the context and the query.
    # Print that prompt to the screen so we can see it.
    # Feed that prompt into the LLM.
    # Format the poutput from the LLM so we can print it.
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | print_input
    | llm
    | StrOutputParser()
)

# This is where we actually invoke the chain with our query.
# By doing it like this we have one chain that is reusable with different queries.
print_with_time("Invoking chain with query...")
answer = chain.invoke(query)
print()
print_with_time(f"Answer: {answer}")

# The LLM will generate a different answer each time it is called it, even with the exact same prompt.
# This is because the LLM is a probabilistic model and generates text based on probabilities.

# Next we can test the given answer to a 'gold standard' answer.
# There are multiple ways this could be done. 

# Test the answer against a standard answer using cosine similarity.
# Here we use another AI to do the similarity testing (AI number 3 in this example).
print_with_time("Loading SentenceTransformer model for similarity testing...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Our gold standard answer.
standard_answer = "The Riley Nine Brooklands was a car produced by Riley Motors between 1928 and 1932."
#standard_answer = "The Riley Nine combustion chambers are hemispherical."
print()

# Compute the cosine similarity between the given answer and the standard answer.
# As with our data store we convert our answers into vectors so we can compare them
print_with_time("Computing cosine similarity between the given answer and the standard answer...")
print_with_time(f"Standard answer: {standard_answer}")
print_with_time(f"Given answer: {answer}")

embeddings1 = model.encode([answer], convert_to_tensor=True)
embeddings2 = model.encode([standard_answer], convert_to_tensor=True)
cosine_similarities = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
print_with_time(f"Similarity score: {cosine_similarities.item()}")

# We are given a score between 0 and 1, where 1 is identical.
# We can set a threshold to determine if the given answer is sufficiently similar to the standard answer.
# And we either pass or fail based on that threshold and our answer.
threshold = 0.80
try:
    similarity_score = cosine_similarities.item()
    assert similarity_score > threshold, "The given answer is not sufficiently similar to the gold standard answer."
    print_with_time(f"\033[92mTest passed: The given answer score of {similarity_score:.2f} is sufficiently similar to the gold standard answer.\033[0m")
except AssertionError as e:
    print_with_time(f"\033[91mTest failed: {e}\033[0m")
    print_with_time(f"Similarity was {similarity_score:.2f} but should be > {threshold}")

 # Note: This example does only a single query once and tests the results.
 # A usable test would have to have many queries, called many times over to build up an average result.
