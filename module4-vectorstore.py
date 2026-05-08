import os
import warnings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

warnings.filterwarnings("ignore")

pdf_path = "TransformerAttentionMechanism.pdf"

print(f"Loading PDF: {pdf_path}")
loader = PyPDFLoader(pdf_path)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(pages)
print(f"Created {len(chunks)} chunks from {len(pages)} pages\n")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
sample_embedding = embeddings.embed_query("Hello world")
print(f"Embedding dimensions: {len(sample_embedding)}")
print(f"First 5 values: {sample_embedding[:5]}")

persist_directory = "./chroma_db"
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name="pdf_collection"
)

print(f"Stored {len(chunks)} chunks in ChromaDB!")
print(f"Data saved to: {os.path.abspath(persist_directory)}")