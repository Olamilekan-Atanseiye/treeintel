# -*- coding: utf-8 -*-
"""
Build the species knowledge vector store.
=================================================
Reads every mapped .docx file in knowledge_docs/, tags each chunk with its
CNN class name, embeds them, and persists a Chroma index that the Flask
app's /knowledge routes query at request time.

Run this once, and again any time you add or edit a document:

    python scripts/build_knowledge_base.py

Requires GROQ_API_KEY only if you later call the LLM -- this script itself
only does loading/chunking/embedding, no LLM calls, so it works without any
API key set.
"""

import json
import sys
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "knowledge_docs"
MAP_PATH = BASE_DIR / "species_map.json"
PERSIST_DIR = BASE_DIR / "knowledge_base" / "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    if not MAP_PATH.exists():
        sys.exit(f"species_map.json not found at {MAP_PATH}")

    species_map = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    species_map.pop("_comment", None)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    skipped = []
    loaded = []

    for class_name, filename in species_map.items():
        if not filename:
            skipped.append((class_name, "no document mapped"))
            continue

        doc_path = DOCS_DIR / filename
        if not doc_path.exists():
            skipped.append((class_name, f"file not found: {filename}"))
            continue

        try:
            loader = Docx2txtLoader(file_path=str(doc_path))
            docs = loader.load()
        except Exception as e:
            skipped.append((class_name, f"failed to load: {e}"))
            continue

        for doc in docs:
            doc.metadata["species"] = class_name
            doc.metadata["source_file"] = filename

        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)
        loaded.append((class_name, filename, len(chunks)))

    print("=" * 60)
    print(f"Loaded {len(loaded)} species documents, {len(all_chunks)} chunks total")
    for class_name, filename, n in loaded:
        print(f"  ✓ {class_name:40s} <- {filename}  ({n} chunks)")

    if skipped:
        print("-" * 60)
        print(f"Skipped {len(skipped)} classes:")
        for class_name, reason in skipped:
            print(f"  - {class_name:40s} {reason}")
    print("=" * 60)

    if not all_chunks:
        sys.exit("No chunks were produced -- check knowledge_docs/ and species_map.json")

    print(f"Embedding with {EMBEDDING_MODEL} ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    PERSIST_DIR.parent.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    print(f"✅ Vector store built and persisted to {PERSIST_DIR}")


if __name__ == "__main__":
    main()
