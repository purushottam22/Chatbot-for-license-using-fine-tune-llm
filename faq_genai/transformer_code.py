import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


# Step 1: Extract text from PDF
def load_pdfs_from_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory_path, filename)
            loader = PyMuPDFLoader(filepath)
            documents.extend(loader.load())
    return documents


# Directory path to your PDFs
directory_path = "C:/Users/kumar/PycharmProjects/firstIdea/faq_genai"

# Load and split the documents
documents = load_pdfs_from_directory(directory_path)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
document_chunks = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(document_chunks, embeddings)

# Use the first few chunks as context
context_chunks = "\n\n".join([chunk.page_content for chunk in document_chunks[:4]])


# Load the t5-base-question-generator model
# model_name = "iarfmoose/t5-base-question-generator"
model_name = "t5-11B"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Your answer and context
answer = "generate frequently asked question with answer for given context "


# Concatenate answer and context
input_text = f"<answer> {answer} <context> {context_chunks}"

# Encode input text
input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

# Generate question
with torch.no_grad():
    output = model.generate(input_ids)
    generated_question = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated question:\n", generated_question)
