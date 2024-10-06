import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


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


# Load the tokenizer and model
model_id = "gpt2-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Create a pipeline for text generation
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=500)

# Define the prompt manually
def create_prompt(context):
    return f"Based on the below context, generate comprehensive FAQ questions and answers:\n\n{context}. If you do not know then do not add anything."

# Generate FAQs based on the context
def generate_faq(context):
    prompt_text = create_prompt(context)
    response = pipe(prompt_text, max_length=500, num_return_sequences=1)
    return response[0]["generated_text"]

# Example usage
response = generate_faq(context_chunks)
print(response)