# PDF Q&A Chatbot

Streamlit app for asking questions about uploaded PDFs using LangChain, Chroma, Groq, and Hugging Face embeddings.

## Deploy on Streamlit Cloud

1. Push this repository to GitHub.
2. Go to Streamlit Cloud and create a new app.
3. Select the repository, branch, and set the main file to `app.py`.
4. Add your Groq API key in Streamlit Cloud secrets:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

5. Deploy.

## Local run

```bash
streamlit run app.py
```

If you want the simpler version, run `apppd.py` instead.