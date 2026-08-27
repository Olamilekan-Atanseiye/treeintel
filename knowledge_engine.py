# -*- coding: utf-8 -*-
"""
Knowledge Engine — RAG + LLM layer for IITA TreeAI
=================================================
Given a CNN-predicted species class name, this module:
  1. Retrieves relevant passages from the per-species vector store
     (built by scripts/build_knowledge_base.py)
  2. Generates a structured one-time "overview" (cached to disk), used to
     populate the AI Knowledge panel right after a prediction
  3. Answers free-form follow-up questions about that same species,
     scoped to its documents only (used by the chat box under the panel)

Note on design: the original prototype used LangChain's
create_stuff_documents_chain / create_retrieval_chain helpers. Those live in
a package (langchain_classic) that has moved around across recent LangChain
versions. To keep this stable, retrieval and prompting are done manually
here (retriever -> format context -> single llm.invoke call) -- fewer
moving parts, easier to debug, same underlying idea.
"""

import json
import os
import re
from pathlib import Path
import time

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
SPECIES_MAP_PATH = BASE_DIR / "species_map.json"
PERSIST_DIR = BASE_DIR / "knowledge_base" / "chroma_db"
CACHE_DIR = BASE_DIR / "knowledge_base" / "overview_cache"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

OVERVIEW_PROMPT = """You are a botanical assistant writing for a tropical African tree species \
identification tool. Using ONLY the context below, produce a JSON object with these exact keys:

- "description": one paragraph, general botanical description
- "uses": ethnobotanical / traditional / commercial uses
- "ecology": ecological role and habitat
- "conservation": conservation status, if mentioned

For any section not covered by the context, use exactly this value: \
"Not documented in available sources." Do not invent or infer information that is not present \
in the context. Respond with ONLY the JSON object, no other text.

Species: {species}

Context:
{context}
"""

ASK_PROMPT = """You are a botanical assistant answering a question about a specific tropical \
African tree species, for a field researcher using an identification tool. Using ONLY the \
context below, answer the question directly and concisely. If the context does not contain \
the answer, say so plainly rather than guessing.

Species: {species}

Context:
{context}

Question:
{question}
"""


# =====================================================
# LAZY SINGLETONS
# =====================================================

_embeddings = None
_vectorstore = None
_llm = None
_species_map = None


def _get_species_map():
    global _species_map
    if _species_map is None:
        data = json.loads(SPECIES_MAP_PATH.read_text(encoding="utf-8"))
        data.pop("_comment", None)
        _species_map = data
    return _species_map


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def _get_vectorstore():
    global _vectorstore

    if _vectorstore is None:
        if not PERSIST_DIR.exists():
            return None

        try:
            from langchain_chroma import Chroma

            _vectorstore = Chroma(
                persist_directory=str(PERSIST_DIR),
                embedding_function=_get_embeddings(),
            )

        except Exception as e:
            raise RuntimeError(
                f"Could not open the Chroma vector store: {e}"
            ) from e

    return _vectorstore


def _get_llm():
    global _llm
    if _llm is None:
        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is not set (check your .env file)")
        from langchain_groq import ChatGroq
        _llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    return _llm


# =====================================================
# STATUS / AVAILABILITY
# =====================================================

def knowledge_base_ready() -> bool:
    """True once the vector store has been built at least once."""
    return PERSIST_DIR.exists()


def has_documentation(species: str) -> bool:
    """True if this species is mapped to a real document (not null)."""
    return bool(_get_species_map().get(species))


# =====================================================
# RETRIEVAL
# =====================================================

def _retrieve(species: str, query: str, k: int = 3):
    start = time.time()

    vectorstore = _get_vectorstore()

    if vectorstore is None:
        return []

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": k,
            "filter": {
                "species": species
            }
        }
    )

    docs = retriever.invoke(query)

    elapsed = time.time() - start

    print(
        f"[RAG] Retrieval took {elapsed:.2f} seconds"
    )

    return docs

def _format_context(docs) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


# =====================================================
# OVERVIEW (cached, generated once per species)
# =====================================================

def _safe_cache_key(species: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", species).strip("_")


def _cache_path(species: str) -> Path:
    return CACHE_DIR / f"{_safe_cache_key(species)}.json"


def get_overview(species: str, force_refresh: bool = False) -> dict:
    """
    Returns a structured overview dict for a species:
        { description, uses, ecology, conservation }
    Cached to disk after first generation. Raises RuntimeError with a clear
    message if the knowledge base isn't built or the species has no docs --
    callers (Flask routes) should catch this and respond gracefully rather
    than crashing.
    """
    if not has_documentation(species):
        raise LookupError(f"No documentation is mapped for '{species}' in species_map.json")

    if not knowledge_base_ready():
        raise RuntimeError("Knowledge base not built yet. Run scripts/build_knowledge_base.py")

    cache_file = _cache_path(species)
    if cache_file.exists() and not force_refresh:
        return json.loads(cache_file.read_text(encoding="utf-8"))

    docs = _retrieve(species, query=f"Overview of {species}", k=6)
    if not docs:
        raise LookupError(f"No retrievable content found for '{species}'")

    prompt = OVERVIEW_PROMPT.format(species=species, context=_format_context(docs))
    llm = _get_llm()
    response = llm.invoke(prompt)
    raw_text = response.content if hasattr(response, "content") else str(response)

    overview = _parse_overview_json(raw_text)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(overview, indent=2), encoding="utf-8")

    return overview


def _parse_overview_json(raw_text: str) -> dict:
    cleaned = re.sub(r"^```(json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fall back to a single description field rather than failing outright
        data = {"description": cleaned}

    fallback = "Not documented in available sources."
    return {
        "description": data.get("description", fallback),
        "uses": data.get("uses", fallback),
        "ecology": data.get("ecology", fallback),
        "conservation": data.get("conservation", fallback),
    }


# =====================================================
# Q&A (not cached — scoped to one species per call)
# =====================================================

def ask(species: str, question: str, k: int = 4) -> str:
    if not has_documentation(species):
        return "No documentation is available for this species yet, so I can't answer questions about it."

    if not knowledge_base_ready():
        raise RuntimeError("Knowledge base not built yet. Run scripts/build_knowledge_base.py")

    docs = _retrieve(species, query=question, k=k)
    if not docs:
        return "I couldn't find anything relevant in the available documentation to answer that."

    prompt = ASK_PROMPT.format(species=species, context=_format_context(docs), question=question)
    llm = _get_llm()
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def warm_up():
    """
    Preload the RAG components so the first user request
    does not have to load the embedding model and ChromaDB.
    """
    print("Warming up knowledge engine...")

    _get_species_map()
    _get_embeddings()
    _get_vectorstore()

    print("Knowledge engine ready.")