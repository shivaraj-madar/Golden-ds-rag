

# Suppress ChromaDB telemetry noise before any imports
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"]     = "False"

import sys
import pathlib
from dotenv import load_dotenv

# Load .env from the script's own folder — works from any terminal location
_script_dir = pathlib.Path(__file__).parent.resolve()
_env_path   = _script_dir / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHROMA_PATH     = str(_script_dir / "chroma_db")
COLLECTION_NAME = "rag_videos"
EMBED_MODEL     = "all-MiniLM-L6-v2"
DEFAULT_TOP_K   = 4
GROQ_MODEL = "openai/gpt-oss-120b"   # free, fast, accurate

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a precise teaching assistant answering questions about neural networks
and deep learning.

Rules:
- Answer using ONLY the transcript excerpts provided.
- Always cite the source video title and timestamp for every key claim.
- If the answer is not in the excerpts, say so clearly.
- Do NOT use outside knowledge beyond the excerpts.
- Be clear and concise. Aim for 3-6 sentences.
"""

# ---------------------------------------------------------------------------
# Lazy clients
# ---------------------------------------------------------------------------
_collection  = None
_groq_client = None


def _get_collection():
    global _collection
    if _collection is None:
        embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        client   = chromadb.PersistentClient(path=CHROMA_PATH)
        try:
            _collection = client.get_collection(
                COLLECTION_NAME, embedding_function=embed_fn
            )
        except Exception:
            console.print(
                "\n[red]ChromaDB collection not found.[/red]\n"
                "Run [cyan]python ingest.py[/cyan] first.\n"
            )
            sys.exit(1)
    return _collection


def _get_groq():
    global _groq_client
    if _groq_client is not None:
        return _groq_client

    # Auto-install groq if missing
    try:
        from groq import Groq
    except ImportError:
        console.print("[yellow]Installing groq library...[/yellow]")
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "groq", "-q"],
            timeout=120,
        )
        from groq import Groq

    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key or not api_key.startswith("gsk_"):
        console.print(Panel(
            "[red]GROQ_API_KEY is missing or incorrect.[/red]\n\n"
            "Get your FREE key in 2 minutes:\n\n"
            "  1. Go to [bold]https://console.groq.com[/bold]\n"
            "  2. Sign up with Google / GitHub (free, no credit card)\n"
            "  3. Click [bold]API Keys[/bold]  →  [bold]Create API Key[/bold]\n"
            "  4. Copy the key  (starts with [green]gsk_[/green])\n"
            "  5. Open [cyan].env[/cyan] and add this line:\n\n"
            "     [green]GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx[/green]\n\n"
            "  6. Save .env and run [cyan]python rag.py[/cyan] again",
            title="Groq API Key Missing",
            border_style="red",
            expand=False,
        ))
        sys.exit(1)

    _groq_client = Groq(api_key=api_key)
    console.print(f"[dim]LLM: Groq / {GROQ_MODEL} (free tier)[/dim]")
    return _groq_client


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
def _generate(prompt: str) -> str:
    client = _get_groq()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ===========================================================================
# Core RAG function  (imported by evaluate.py)
# ===========================================================================
def rag_query(question: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """
    Retrieve relevant chunks from ChromaDB and generate an answer with Groq.

    Returns:
        question         : str
        answer           : str
        sources          : list[str]   "video title @ timestamp"
        retrieved_chunks : list of (text, metadata, distance) tuples
    """
    collection = _get_collection()

    # 1. Retrieve top-k chunks
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    docs   = results["documents"][0]
    metas  = results["metadatas"][0]
    dists  = results["distances"][0]

    # 2. Build context string for the LLM
    parts = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        relevance = round((1 - dist) * 100)
        parts.append(
            f"[Excerpt {i+1}]\n"
            f"Source   : {meta['source']} ({meta['channel']})\n"
            f"Timestamp: {meta['timestamp']}\n"
            f"Relevance: {relevance}%\n"
            f"Text     : {doc}"
        )
    context = "\n\n---\n\n".join(parts)
    prompt  = f"Transcript excerpts:\n\n{context}\n\nQuestion: {question}"

    # 3. Generate answer
    answer = _generate(prompt)

    # 4. Deduplicated source list
    seen, sources = set(), []
    for meta in metas:
        key = f"{meta['source']} @ {meta['timestamp']}"
        if key not in seen:
            seen.add(key)
            sources.append(key)

    return {
        "question":         question,
        "answer":           answer,
        "sources":          sources,
        "retrieved_chunks": list(zip(docs, metas, dists)),
    }


# ===========================================================================
# Pretty print
# ===========================================================================
def print_result(result: dict) -> None:
    console.print(Panel.fit(
        f"[bold]Q:[/bold] {result['question']}",
        border_style="blue",
    ))
    console.print("\n[bold]Answer:[/bold]")
    console.print(Markdown(result["answer"]))
    console.print("\n[bold]Retrieved from:[/bold]")
    for s in result["sources"]:
        console.print(f"  [cyan]•[/cyan] {s}")
    console.print()


# ===========================================================================
# Smoke test
# ===========================================================================
if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold]RAG Smoke Test[/bold]  [dim](Groq / Llama3 — free)[/dim]",
        border_style="blue",
    ))

    question = "What is the difference between a weight and a bias in a neural network?"
    console.print(f"\nTest question: [italic]{question}[/italic]\n")
    
    result = rag_query(question)
    print_result(result)

    console.print(
        "[green]Smoke test passed![/green]  "
        "Run next: [cyan]python evaluate.py[/cyan]\n"
    )