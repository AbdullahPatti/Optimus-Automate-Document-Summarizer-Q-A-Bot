# Document Summarizer & Q&A Bot

A simple Streamlit app that accepts PDF or text documents, generates summaries, and answers questions using retrieval-augmented generation (RAG).

## Setup

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Run the app:

```bash
streamlit run app.py
```

3. Enter your OpenAI API key in the sidebar or set `OPENAI_API_KEY` in your environment.

## Usage

- Upload a PDF or TXT file.
- The app extracts text, splits it into chunks, computes embeddings or uses TF-IDF, and builds an index.
- It generates a document summary and lets you ask questions about the content.

## Notes

- Use `openai` provider when you have an OpenAI API key.
- Use `openrouter` provider with an OpenRouter API key and model like `openai/gpt-oss-120b:free`.
- Use `local-tfidf` for a keyless fallback; it returns a simple summary and the most relevant document excerpt.
- For better retrieval with OpenAI, choose `text-embedding-3-large` and `gpt-4o-mini`.
"# Optimus-Automate-Document-Summarizer-Q-A-Bot" 
