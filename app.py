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

if "num_pages" not in st.session_state:
    st.session_state["num_pages"] = 0

if "num_chunks" not in st.session_state:
    st.session_state["num_chunks"] = 0

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

st.markdown(
    """
    <style>
    :root {
        --bg: #f7f8fc;
        --ink: #121826;
        --muted: #5a6475;
        --card: #ffffff;
        --line: #e5e7ef;
        --accent: #0f766e;
        --accent-2: #f59e0b;
        --accent-3: #0ea5e9;
    }

    .stApp {
        background:
            radial-gradient(1200px 400px at 10% -10%, #d9f4ff 0%, transparent 60%),
            radial-gradient(1000px 500px at 100% 0%, #fff1d6 0%, transparent 60%),
            var(--bg);
    }

    .hero-wrap {
        background: linear-gradient(120deg, #0f766e, #0ea5e9);
        color: white;
        border-radius: 20px;
        padding: 1.2rem 1.3rem;
        margin: 0.2rem 0 1rem 0;
        box-shadow: 0 10px 30px rgba(14, 32, 64, 0.15);
        animation: fadeUp 0.55s ease;
    }

    .hero-title {
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: 0.2px;
        margin: 0;
    }

    .hero-sub {
        margin-top: 0.35rem;
        font-size: 0.97rem;
        opacity: 0.92;
    }

    .chip-row {
        display: flex;
        gap: 0.45rem;
        flex-wrap: wrap;
        margin-top: 0.85rem;
    }

    .chip {
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 999px;
        padding: 0.2rem 0.65rem;
        font-size: 0.78rem;
        backdrop-filter: blur(3px);
    }

    .stat-card {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.8rem 0.85rem;
        box-shadow: 0 8px 22px rgba(16, 24, 40, 0.06);
        animation: fadeUp 0.55s ease;
    }

    .stat-label {
        color: var(--muted);
        font-size: 0.8rem;
        margin-bottom: 0.15rem;
    }

    .stat-value {
        color: var(--ink);
        font-size: 1.2rem;
        font-weight: 800;
    }

    .guide-card {
        background: white;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 0.95rem;
        min-height: 120px;
        animation: fadeUp 0.6s ease;
    }

    .guide-k {
        color: var(--accent);
        font-weight: 700;
        font-size: 0.86rem;
    }

    .guide-h {
        color: var(--ink);
        font-size: 1.05rem;
        font-weight: 800;
        margin: 0.2rem 0;
    }

    .guide-p {
        color: var(--muted);
        font-size: 0.9rem;
        margin: 0;
    }

    .stChatMessage {
        border-radius: 14px;
    }

    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

st.markdown(
    """
    <div class="hero-wrap">
        <p class="hero-title">PDF Q&A Chatbot</p>
        <p class="hero-sub">Ask smart questions from your document with a clean RAG workflow.</p>
        <div class="chip-row">
            <span class="chip">Groq</span>
            <span class="chip">LangChain</span>
            <span class="chip">ChromaDB</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Control Panel")
    st.caption("Upload your PDF and manage session state")

    uploaded_file = st.file_uploader(
        "Drag and drop your pdf here",
        type=["pdf"],
        help="Upload a pdf file to start asking questions about it"
    )

    st.markdown("---")
    if st.session_state.pdf_processed:
        st.success("Document is ready")
    else:
        st.info("Waiting for upload")

if uploaded_file and not st.session_state.pdf_processed:
    vector_store, num_pages, num_chunks = process_pdf(uploaded_file)

    st.session_state.vector_store = vector_store
    st.session_state.pdf_processed = True
    st.session_state.chat_history = []
    st.session_state.num_pages = num_pages
    st.session_state.num_chunks = num_chunks

    st.success(f"Processed: {num_pages} pages → {num_chunks} chunks")

if st.session_state.pdf_processed:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Pages</div>
                <div class="stat-value">{st.session_state.num_pages}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Chunks</div>
                <div class="stat-value">{st.session_state.num_chunks}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Messages</div>
                <div class="stat-value">{len(st.session_state.chat_history)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

if st.session_state.pdf_processed:

    if st.button("Upload New PDF"):
        st.session_state.pdf_processed = False
        st.session_state.vector_store = None
        st.session_state.chat_history = []
        st.session_state.num_pages = 0
        st.session_state.num_chunks = 0
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
        st.markdown(
            """
            <div class="guide-card">
                <p class="guide-k">Step 1</p>
                <p class="guide-h">Upload</p>
                <p class="guide-p">Drop your PDF in the left panel to begin.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="guide-card">
                <p class="guide-k">Step 2</p>
                <p class="guide-h">Ask</p>
                <p class="guide-p">Type natural language questions in chat.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="guide-card">
                <p class="guide-k">Step 3</p>
                <p class="guide-h">Answer</p>
                <p class="guide-p">Get concise responses grounded in your document.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        