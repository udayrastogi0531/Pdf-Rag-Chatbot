# 📄 PDF Q&A Chatbot — RAG Powered AI

![Banner](banner.png)

## ✨ Overview
Welcome to the **PDF Q&A Chatbot**, a state-of-the-art Retrieval Augmented Generation (RAG) application. Upload any PDF and start chatting with it! This app uses advanced LLMs and Vector Embeddings to provide precise, context-aware answers directly from your documents.

---

## 🚀 Key Features
- **⚡ Super-Fast Inference**: Powered by **Groq (Llama 3.3)** for near-instant responses.
- **🧠 Smart Retrieval**: Uses **ChromaDB** and **HuggingFace Embeddings** to find exactly what you're looking for.
- **🎨 Premium UI**: A sleek, modern Streamlit interface with custom CSS and interactive stats.
- **📊 Document Insights**: Real-time stats on page counts, text chunks, and message history.
- **🛡️ Source Tracking**: View exactly which parts of the PDF were used to generate the answer.
- **🔄 Session Memory**: Maintains a conversational history for a natural chat experience.

---

## 🛠️ Tech Stack
| Component | Technology |
| :--- | :--- |
| **Frontend** | [Streamlit](https://streamlit.io/) |
| **Orchestration** | [LangChain](https://www.langchain.com/) |
| **LLM** | [Groq (Llama 3.3-70B)](https://groq.com/) |
| **Vector DB** | [ChromaDB](https://www.trychroma.com/) |
| **Embeddings** | [HuggingFace (all-MiniLM-L6-v2)](https://huggingface.co/) |

---

## 🏗️ How It Works (RAG Workflow)
1. **Upload**: You drop a PDF into the sidebar.
2. **Chunking**: The app splits the PDF into smaller, manageable text chunks.
3. **Embedding**: Each chunk is converted into a mathematical vector representation.
4. **Storage**: Vectors are stored in **ChromaDB** for fast similarity searching.
5. **Querying**: When you ask a question, the app finds the most relevant chunks.
6. **Generation**: Relevant chunks + your question are sent to the LLM to produce a grounded answer.

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/udayrastogi0531/Pdf-Rag-Chatbot.git
cd Pdf-Rag-Chatbot
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/scripts/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory and add your Groq API Key:
```env
GROQ_API_KEY=your_actual_key_here
```

### 5. Run the App
```bash
streamlit run app.py
```

---

## 🌐 Deployment Guide (Streamlit Cloud)

1. **Push** your code to GitHub.
2. Go to **[share.streamlit.io](https://share.streamlit.io/)** and create a new app.
3. Select your repository and set `app.py` as the entry point.
4. **Crucial Step**: Add your API Key in **Secrets**:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
5. Deploy and enjoy!

---

## 🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

## 📄 License
This project is licensed under the MIT License.

---
Created with ❤️ by [Uday Rastogi](https://github.com/udayrastogi0531)