import io
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import pdfplumber
import streamlit as st
from openai import OpenAI
from openrouter import OpenRouter, errors as openrouter_errors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_openai_client(api_key: Optional[str]) -> OpenAI:
    if api_key:
        return OpenAI(api_key=api_key)
    return OpenAI()


def get_openrouter_client(api_key: str) -> OpenRouter:
    return OpenRouter(api_key=api_key)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_pages.append(page_text)
    return "\n\n".join(text_pages)


def extract_text(file) -> str:
    file.seek(0)
    if file.type == "application/pdf" or file.name.lower().endswith(".pdf"):
        import io

        return extract_text_from_pdf(file.read())
    raw_bytes = file.read()
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("latin-1", errors="ignore")


def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_length:
            break
        start = end - overlap
    return chunks


def get_embeddings(texts: List[str], api_key: Optional[str], model: str = "text-embedding-3-small") -> np.ndarray:
    client = get_openai_client(api_key)
    response = client.embeddings.create(model=model, input=texts)
    return np.array([item.embedding for item in response.data], dtype=np.float32)


def build_document_index(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    provider: str,
    api_key: Optional[str],
    embed_model: str,
) -> Tuple[List[str], np.ndarray, Optional[TfidfVectorizer]]:
    chunks = split_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
    if not chunks:
        return [], np.zeros((0, 0), dtype=np.float32), None

    if provider == "openai" and api_key:
        embeddings = get_embeddings(chunks, api_key, model=embed_model)
        return chunks, embeddings, None

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(chunks)
    return chunks, tfidf_matrix, vectorizer


def cosine_similarities(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_emb)
    doc_norms = np.linalg.norm(doc_embs, axis=1)
    if query_norm == 0 or np.any(doc_norms == 0):
        return np.zeros(doc_embs.shape[0], dtype=np.float32)
    normalized_query = query_emb / query_norm
    normalized_docs = doc_embs / doc_norms[:, None]
    return np.dot(normalized_docs, normalized_query)


def retrieve_top_chunks(
    query: str,
    chunks: List[str],
    embeddings: np.ndarray,
    api_key: Optional[str],
    embed_model: str,
    top_k: int,
    vectorizer: Optional[TfidfVectorizer] = None,
) -> List[Tuple[str, float]]:
    if not chunks:
        return []

    if vectorizer is not None:
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, embeddings).flatten()
        best_indices = np.argsort(similarities)[::-1][: min(top_k, len(chunks))]
        return [(chunks[idx], float(similarities[idx])) for idx in best_indices]

    query_emb = get_embeddings([query], api_key, model=embed_model)[0]
    similarities = cosine_similarities(query_emb, embeddings)
    best_indices = np.argsort(similarities)[::-1][: min(top_k, len(chunks))]
    return [(chunks[idx], float(similarities[idx])) for idx in best_indices]


def send_openrouter_chat(prompt: str, model: str, api_key: str, max_tokens: int = 400) -> str:
    for attempt in range(3):
        try:
            with get_openrouter_client(api_key) as client:
                response = client.chat.send(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes text accurately."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
            return response.choices[0].message.content.strip()
        except openrouter_errors.TooManyRequestsResponseError as exc:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(
                "OpenRouter rate limit exceeded. Please wait a moment and retry, or switch providers."
            ) from exc
        except openrouter_errors.OpenRouterError as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc


def summarize_text(
    chunks: List[str],
    provider: str,
    api_key: Optional[str],
    response_model: str,
    max_tokens: int = 400,
) -> str:
    if provider == "local-tfidf":
        summary_chunks = chunks[: min(3, len(chunks))]
        return "\n\n".join(summary_chunks)

    prompt = (
        "You are a document summarization assistant. Read the provided excerpts and produce a concise summary that captures the key points, "
        "main conclusions, and any important data or findings. Keep the response short, accurate, and organized.\n\n"
        "EXCERPTS:\n"
        + "\n\n---\n\n".join(chunks)
        + "\n\nSUMMARY:"
    )
    if provider == "openrouter":
        return send_openrouter_chat(prompt, response_model, api_key or "", max_tokens=max_tokens)

    client = get_openai_client(api_key)
    response = client.responses.create(
        model=response_model,
        input=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text accurately."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response.output_text.strip()


def answer_question(
    question: str,
    context_chunks: List[Tuple[str, float]],
    provider: str,
    api_key: Optional[str],
    response_model: str,
    max_tokens: int = 400,
) -> str:
    if provider == "local-tfidf":
        best_chunk, _ = context_chunks[0]
        return (
            "The best document excerpt for your question is below. "
            "For a full answer you need an external model, but this text is the most relevant excerpt.\n\n"
            + best_chunk
        )

    context_text = "\n\n---\n\n".join([chunk for chunk, _ in context_chunks])
    prompt = (
        "You are an expert assistant answering questions from a document. Use only the information provided in the document excerpts. "
        "If the answer is not contained in the excerpts, say that the document does not contain the answer.\n\n"
        "DOCUMENT EXCERPTS:\n"
        + context_text
        + "\n\nQUESTION:\n"
        + question
        + "\n\nANSWER:"
    )
    if provider == "openrouter":
        return send_openrouter_chat(prompt, response_model, api_key or "", max_tokens=max_tokens)

    client = get_openai_client(api_key)
    response = client.responses.create(
        model=response_model,
        input=[
            {"role": "system", "content": "You are a precise, conservative assistant writing factual answers."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response.output_text.strip()


def main() -> None:
    st.set_page_config(page_title="Document Summarizer & Q&A", layout="wide")
    st.title("Document Summarizer & Q&A Bot")
    st.write(
        "Upload a PDF or plain text file, generate an automatic summary, and ask questions about the document using retrieval-augmented generation."
    )

    with st.sidebar:
        st.header("Configuration")
        provider = st.selectbox(
            "Provider",
            ["openai", "openrouter", "local-tfidf"],
            index=0,
            help="Choose the provider for model and retrieval. local-tfidf uses a built-in fallback when no external API is available.",
        )
        api_key = st.text_input("API Key", type="password", help="OpenAI or OpenRouter API key depending on provider.")
        if provider in ["openai", "openrouter"]:
            response_model = st.text_input(
                "Response Model",
                value="gpt-4o-mini" if provider == "openai" else "openai/gpt-oss-120b:free",
                help="Set the model name for chat/summarization.",
            )
        else:
            response_model = st.text_input(
                "Response Model",
                value="openai/gpt-oss-120b:free",
                disabled=True,
                help="Local TF-IDF uses only the built-in retrieval and cannot call external chat models.",
            )
        embed_model = st.selectbox(
            "Embeddings Model",
            ["text-embedding-3-large", "text-embedding-3-small"],
            index=0,
            help="Choose the embedding model for retrieval accuracy.",
        )
        chunk_size = st.slider("Chunk size", min_value=600, max_value=2000, value=1000, step=100)
        chunk_overlap = st.slider("Chunk overlap", min_value=100, max_value=500, value=200, step=50)
        top_k = st.slider("Top retrieval chunks", min_value=1, max_value=8, value=4, step=1)

    uploaded_file = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt"])
    if not uploaded_file:
        st.warning("Please upload a PDF or TXT document to continue.")
        return

    text = extract_text(uploaded_file)
    if not text.strip():
        st.error("Could not extract any text from the uploaded file.")
        return

    st.success("Document uploaded successfully.")
    st.write(f"Document length: **{len(text)} characters**")

    with st.expander("Document preview", expanded=False):
        st.text_area("Document text preview", value=text[:6000], height=300)

    if provider in ["openai", "openrouter"] and not api_key:
        st.warning("Enter your OpenAI or OpenRouter API key in the sidebar.")
        return

    auth_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if api_key:
        if provider == "openai":
            os.environ["OPENAI_API_KEY"] = api_key
        elif provider == "openrouter":
            os.environ["OPENROUTER_API_KEY"] = api_key

    with st.spinner("Building document embeddings and indexing content..."):
        chunks, embeddings, vectorizer = build_document_index(text, chunk_size, chunk_overlap, provider, auth_key, embed_model)

    st.write(f"Document split into **{len(chunks)} chunks** for retrieval.")

    summary = None
    try:
        summary_chunks = chunks[: min(8, len(chunks))]
        summary = summarize_text(summary_chunks, provider, auth_key, response_model)
    except Exception as exc:
        st.error(f"Unable to generate summary: {exc}")

    if summary:
        st.header("Document Summary")
        st.write(summary)

    if "qna_history" not in st.session_state:
        st.session_state.qna_history = []
    if "question_input" not in st.session_state:
        st.session_state.question_input = ""

    def clear_question() -> None:
        st.session_state.question_input = ""

    st.header("Ask a question")
    question = st.text_input("Enter a question about the document", key="question_input")
    ask = st.button("Get answer")
    st.button("Clear question", on_click=clear_question)

    if ask:
        if not question.strip():
            st.warning("Please type a question before asking.")
        else:
            with st.spinner("Retrieving relevant content and generating an answer..."):
                context_chunks = retrieve_top_chunks(
                    question,
                    chunks,
                    embeddings,
                    auth_key,
                    embed_model,
                    top_k,
                    vectorizer=vectorizer,
                )
                if not context_chunks:
                    st.error("No indexed document content is available for retrieval.")
                else:
                    try:
                        answer = answer_question(
                            question,
                            context_chunks,
                            provider,
                            auth_key,
                            response_model,
                        )
                        st.session_state.qna_history.append(
                            {
                                "question": question,
                                "answer": answer,
                                "context": context_chunks,
                            }
                        )
                    except Exception as exc:
                        st.error(f"Unable to generate answer: {exc}")

    if st.session_state.qna_history:
        st.header("Previous Q&A")
        for idx, item in enumerate(st.session_state.qna_history, start=1):
            st.subheader(f"Q{idx}: {item['question']}")
            st.write(item["answer"])
            with st.expander("Retrieved context snippets", expanded=False):
                for c_idx, (chunk, score) in enumerate(item["context"], start=1):
                    st.markdown(f"**Snippet {c_idx}** (score: {score:.3f})")
                    st.write(chunk)


if __name__ == "__main__":
    main()
