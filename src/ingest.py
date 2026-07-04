"""Charge les documents (PDF, CSV) du dossier data/, les decoupe en chunks,
les vectorise et les insere dans Pinecone."""

import sys
from pathlib import Path

import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_embeddings, get_pinecone_index

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_pdf(path: Path) -> list[Document]:
    return PyPDFLoader(str(path)).load()


def load_csv(path: Path) -> list[Document]:
    df = pd.read_csv(path)
    docs = []
    for i, row in df.iterrows():
        text = ", ".join(f"{col}: {row[col]}" for col in df.columns)
        docs.append(Document(page_content=text, metadata={"source": path.name, "row": i}))
    return docs


def load_all_documents() -> list[Document]:
    docs: list[Document] = []
    for path in DATA_DIR.glob("*"):
        if path.suffix.lower() == ".pdf":
            docs.extend(load_pdf(path))
        elif path.suffix.lower() == ".csv":
            docs.extend(load_csv(path))
    return docs


def main():
    documents = load_all_documents()
    if not documents:
        print(f"Aucun fichier PDF/CSV trouve dans {DATA_DIR}")
        sys.exit(1)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    print(f"{len(documents)} document(s) charge(s) -> {len(chunks)} chunks")

    embeddings = get_embeddings()
    vectors = embeddings.embed_documents([c.page_content for c in chunks])

    index = get_pinecone_index()
    batch = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        batch.append({
            "id": f"chunk-{i}",
            "values": vector,
            "metadata": {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page", chunk.metadata.get("row", 0)),
            },
        })
        if len(batch) == 100:
            index.upsert(vectors=batch)
            batch = []
    if batch:
        index.upsert(vectors=batch)

    print(f"Ingestion terminee : {len(chunks)} chunks indexes dans Pinecone.")


if __name__ == "__main__":
    main()
