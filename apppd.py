import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

# LLM
from langchain_groq import ChatGroq

# Prompt
from langchain_core.prompts import ChatPromptTemplate

# PDF Loading & Chunking
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings & Vector Store
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# RAG Chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# Load API Keys 
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PDF Q&A Chatbot",
    layout="wide"
)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

if "pdf_processed" not in st.session_state:
    st.session_state["pdf_processed"] = False

@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = get_llm()
embeddings = get_embeddings()

rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant that answers questions based ONLY
on the provided context from a PDF document.

RULES:
- Answer the question using ONLY the information in the context below
- Be concise and direct in your response
- If the context contains relevant information, provide a clear answer
- Use bullet points for listing multiple items

IMPORTANT: If the answer is NOT in the provided context, strictly say:
"I don't know based on the provided document."
DO NOT make up or assume any information.

CONTEXT:
{context}"""
    ),
    (
        "human",
        "{input}"
    )
])

def process_pdf(uploaded_file):
    with st.spinner("Reading PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

    with st.spinner("Splitting into chunks..."):
        loader = PyPDFLoader(temp_file_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        chunks = text_splitter.split_documents(pages)

    with st.spinner(f"Creating embeddings for {len(chunks)} chunks..."):
        persist_dir = os.path.join(tempfile.gettempdir(), "Chroma_Streamlit")

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name="streamlit_pdf"
        )

        os.unlink(temp_file_path)

        return vector_store, len(pages), len(chunks)

st.title("PDF Q&A Chatbot")
st.caption("Upload a PDF and ask questions about it! Powered by Groq + LangChain + ChromaDB")

with st.sidebar:
    st.header("Upload PDF")

    uploaded_file = st.file_uploader(
    "Drag and drop your pdf here",
    type=["pdf"],
    help="Upload a pdf file to start asking questions about it"
)

if uploaded_file and not st.session_state.pdf_processed:
    vector_store, num_pages, num_chunks = process_pdf(uploaded_file)

    st.session_state.vector_store = vector_store
    st.session_state.pdf_processed = True
    st.session_state.chat_history = []

    st.success(f"Processed: {num_pages} pages → {num_chunks} chunks")

if st.session_state.pdf_processed:
    st.info("PDF loaded and ready for questions!")

if st.session_state.pdf_processed:

    if st.button("Upload New PDF"):
        st.session_state.pdf_processed = False
        st.session_state.vector_store = None
        st.session_state.chat_history = []
        st.rerun()

        st.divider()

st.markdown("""
### How it works

1. **Upload** a PDF document
2. The app **splits** it into small chunks
3. Chunks are converted to **embeddings**
4. Ask a question → app **searches** for relevant chunks
5. Relevant chunks + question → **LLM** → Answer!

_This is called **RAG** (Retrieval Augmented Generation)_
""")

if st.session_state.pdf_processed:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input(
        "Ask a question about your PDF...",
        disabled=not st.session_state.pdf_processed
    )

    if user_question:

        with st.chat_message("user"):
            st.markdown(user_question)

        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                document_chain = create_stuff_documents_chain(llm, rag_prompt)

                retriever = st.session_state.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                )

                rag_chain = create_retrieval_chain(retriever, document_chain)

                response = rag_chain.invoke({"input": user_question})
                answer = response["answer"]

                st.markdown(answer)

                with st.expander("View Source Chunks"):
                    for i, doc in enumerate(response["context"], 1):
                        page_num = doc.metadata.get("page", "?")
                        st.markdown(f"**Chunk {i}** (Page {page_num}):")
                        st.caption(doc.page_content[:300] + "...")
                        st.divider()

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })

if not st.session_state.pdf_processed:
    st.info("Upload a PDF in the sidebar to get started!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 1 Upload")
        st.markdown("Drag a PDF into the sidebar uploader")

    with col2:
        st.markdown("### 2 Ask")
        st.markdown("Type your questions in the chatbox")

    with col3:
        st.markdown("### 3 Answer")
        st.markdown("Get instant answers from your PDF")

        