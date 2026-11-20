
import os
import re

from pathlib import Path
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Folder containing PHC documents
DOCUMENT_FOLDER = Path(__file__).parent / "phc_documents"


def extract_text(file_path):
    """Extract text from PDF or TXT files."""

    file_path = Path(file_path)

    if file_path.suffix.lower() == ".pdf":
        reader = PdfReader(str(file_path))
        text = ""

        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"

        return text

    elif file_path.suffix.lower() == ".txt":
        return file_path.read_text(encoding="utf-8")

    return ""


def split_into_chunks(text, chunk_size=150, overlap=30):
    """Split document text into smaller sections."""

    words = re.findall(r"\S+", text)

    chunks = []

    for start in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[start:start + chunk_size])

        if chunk.strip():
            chunks.append(chunk)

    return chunks


def load_documents():
    """Read all PDF and TXT documents."""

    all_chunks = []
    sources = []

    DOCUMENT_FOLDER.mkdir(exist_ok=True)

    for file_path in DOCUMENT_FOLDER.iterdir():

        if file_path.suffix.lower() not in [".pdf", ".txt"]:
            continue

        try:
            document_text = extract_text(file_path)

            if not document_text.strip():
                print(f"No readable text in: {file_path.name}")
                continue

            chunks = split_into_chunks(document_text)

            for chunk in chunks:
                all_chunks.append(chunk)
                sources.append(file_path.name)

        except Exception as error:
            print(f"Error reading {file_path.name}: {error}")

    return all_chunks, sources



# The index is rebuilt only when files in phc_documents/ change (not on every
# question), which saves time and avoids re-reading every PDF repeatedly.
_INDEX = {"signature": None, "chunks": [], "sources": [],
          "vectorizer": None, "matrix": None}


def _folder_signature():
    DOCUMENT_FOLDER.mkdir(exist_ok=True)
    return tuple(sorted(
        (f.name, f.stat().st_mtime_ns, f.stat().st_size)
        for f in DOCUMENT_FOLDER.iterdir()
        if f.suffix.lower() in (".pdf", ".txt")))


def _get_index():
    signature = _folder_signature()
    if signature == _INDEX["signature"]:
        return _INDEX
    chunks, sources = load_documents()
    vectorizer = matrix = None
    if chunks:
        try:
            vectorizer = TfidfVectorizer(stop_words="english",
                                         ngram_range=(1, 2), sublinear_tf=True)
            matrix = vectorizer.fit_transform(chunks)
        except ValueError:
            vectorizer = matrix = None
    _INDEX.update(signature=signature, chunks=chunks, sources=sources,
                  vectorizer=vectorizer, matrix=matrix)
    return _INDEX


def search_documents(question, top_k=3):
    """Search all uploaded documents and return the most relevant chunks."""

    index = _get_index()

    if index["vectorizer"] is None:
        return []

    try:
        chunks, sources = index["chunks"], index["sources"]
        question_vector = index["vectorizer"].transform([question])

        scores = cosine_similarity(
            question_vector,
            index["matrix"]
        ).flatten()

        # Sort chunks by relevance score
        ranked_indices = scores.argsort()[::-1]

        results = []
        seen_chunks = set()

        for i in ranked_indices:

            if scores[i] <= 0:
                continue

            chunk_text = chunks[i].strip()

            # Avoid duplicate results
            if chunk_text in seen_chunks:
                continue

            seen_chunks.add(chunk_text)

            results.append({
                "text": chunk_text,
                "source": sources[i],
                "score": round(float(scores[i]), 3)
            })

            if len(results) >= top_k:
                break

        return results

    except ValueError:
        return []


def get_demo_answer(question):
    """Return retrieved content as a temporary demo response."""

    results = search_documents(question)

    if not results:
        return (
            "I could not find relevant information in the uploaded "
            "PHC documents. Please upload a relevant PDF or TXT file."
        )

    answer = "Relevant information found in your PHC documents:\n\n"

    for number, result in enumerate(results, start=1):
        answer += f"{number}. {result['text']}\n"
        answer += f"Source: {result['source']}\n\n"

    answer += (
        "Note: This is a document retrieval demo. "
        "AI-generated answers will be added through the Nexus API."
    )

    return answer
