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

# This script demonstrates a simple feedback generating system with the input being 
# from sample email type comments.

print_with_time("Creating LLM chain...")
llm = OllamaLLM(model="deepseek-r1:7b")

print_with_time("Reading context from RawFeedback.txt...")
with open(".\\FeedbackData\\RawFeedback.txt", "r", encoding="utf-8") as f:
    raw_feedback = f.read()

# Template for the prompt (no question needed)
template = """
Numbered instructions are as follows. Make sure to do them all.
1. From the following pieces of feedback give me a summary of the 5 main points.
2. List the points in order of most common to least common.
3. Do not include any personal information.
4. Do not include any information that is not in the context.
5. Ensure the answer is polite.
6. Only give your feedback in the answer, don't add any extra information or comment.
7. At the end give a general impression of what people think of this application.

Context: {context}
"""
prompt = PromptTemplate.from_template(template)

# Print input for logging
def print_input(input):
    print_with_time(f"Input prompt: {input}")
    return input

# Build the chain using static context from the file
chain = (
    {"context": lambda _: raw_feedback}
    | prompt
    | print_input
    | llm
    | StrOutputParser()
)

# Invoke the chain
print_with_time("Invoking chain with prompt...")
answer = chain.invoke({})
print()
print_with_time(f"Feedback: {answer}")