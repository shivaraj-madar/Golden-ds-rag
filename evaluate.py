"""
evaluate.py
-----------
Runs the 5 Golden QA pairs through the RAG pipeline and prints
a structured evaluation report.

Scoring:
  - Source precision  : Did the top retrieved chunk come from the correct video?
  - Keyword coverage  : What % of expected answer keywords appear in the response?

Run ingest.py and verify rag.py works before running this.

Usage:
    python evaluate.py
"""

import time
from rag import rag_query
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule

import json

with open("data/dataset.json", "r") as f:
    dataset = json.load(f)

console = Console()

# ===========================================================================
# Golden QA dataset — 5 hand-crafted pairs from the 4 assignment videos
# ===========================================================================
GOLDEN_QA = [
    {
        "id": "QA-01",
        "question": (
            "What is the difference between a weight and a bias in a neural "
            "network, and why do we need both?"
        ),
        "expected_source": "But what is a Neural Network?",
        "expected_channel": "3Blue1Brown",
        "expected_keywords": ["weight", "bias", "activation", "threshold", "trainable"],
        "notes": "Tests precision: weight-vs-bias passage, not a gradient descent chunk.",
    },
    {
        "id": "QA-02",
        "question": (
            "What role do hidden layers play when recognising handwritten digits, "
            "according to 3Blue1Brown?"
        ),
        "expected_source": "But what is a Neural Network?",
        "expected_channel": "3Blue1Brown",
        "expected_keywords": ["hidden", "edge", "loop", "layer", "stroke"],
        "notes": "Tests conceptual reasoning about architecture, not training mechanics.",
    },
    {
        "id": "QA-03",
        "question": (
            "Why is softmax applied at the final step of a transformer, "
            "and what are logits?"
        ),
        "expected_source": "Transformers, the tech behind LLMs",
        "expected_channel": "3Blue1Brown",
        "expected_keywords": ["softmax", "logit", "probability", "token", "vocabulary"],
        "notes": "Source disambiguation trap: sigmoid from Video 1 is a wrong retrieval.",
    },
    {
        "id": "QA-04",
        "question": (
            "What core limitation of traditional machine learning does deep learning "
            "address, and how does this relate to the word 'deep'?"
        ),
        "expected_source": "What is Deep Learning?",
        "expected_channel": "CampusX",
        "expected_keywords": ["feature", "manual", "representation", "layer", "automatic"],
        "notes": "Cross-language retrieval test (Hindi source, English question).",
    },
    {
        "id": "QA-05",
        "question": (
            "What is overfitting in machine learning, how does it arise in a "
            "supervised learning model, and why does it hurt real-world performance?"
        ),
        "expected_source": "All About ML & Deep Learning",
        "expected_channel": "CodeWithHarry",
        "expected_keywords": ["overfit", "training", "generalise", "test", "noise"],
        "notes": "Cause-effect reasoning test — definition alone is not enough.",
    },
]


# ===========================================================================
# Scorer
# ===========================================================================
def score_result(result: dict, qa: dict) -> dict:
    """
    Returns a dict with scoring details for one QA pair.
    """
    chunks = result["retrieved_chunks"]

    # Source precision — did the top chunk come from the right video?
    top_source  = chunks[0][1]["source"] if chunks else ""
    source_hit  = qa["expected_source"].lower() in top_source.lower()

    # Also check if ANY of the top-4 chunks came from the right video
    any_source_hit = any(
        qa["expected_source"].lower() in c[1]["source"].lower()
        for c in chunks
    )

    # Keyword coverage in the generated answer
    answer_lower = result["answer"].lower()
    kw_hits      = [kw for kw in qa["expected_keywords"] if kw in answer_lower]
    kw_coverage  = round(len(kw_hits) / len(qa["expected_keywords"]) * 100)

    # Relevance score of top chunk (cosine similarity, 0–100)
    top_relevance = round((1 - chunks[0][2]) * 100) if chunks else 0

    return {
        "source_correct":    source_hit,
        "any_source_hit":    any_source_hit,
        "keyword_coverage":  kw_coverage,
        "keywords_found":    kw_hits,
        "keywords_missed":   [k for k in qa["expected_keywords"] if k not in kw_hits],
        "top_source":        top_source,
        "top_relevance":     top_relevance,
    }


# ===========================================================================
# Report printer
# ===========================================================================
def print_detail(qa: dict, result: dict, scores: dict) -> None:
    src_tag = "[green]✓ CORRECT[/green]" if scores["source_correct"] else (
              "[yellow]~ IN TOP-4[/yellow]" if scores["any_source_hit"] else
              "[red]✗ WRONG[/red]")

    console.print(f"\n  [bold]{qa['id']}[/bold]  {src_tag}  "
                  f"Keywords: [cyan]{scores['keyword_coverage']}%[/cyan]  "
                  f"Top relevance: [dim]{scores['top_relevance']}%[/dim]")
    console.print(f"  [dim]Q: {qa['question'][:80]}...[/dim]")
    console.print(f"  [dim]Retrieved: {scores['top_source'][:60]}[/dim]")
    if scores["keywords_missed"]:
        console.print(f"  [dim]Missing keywords: {scores['keywords_missed']}[/dim]")
    console.print(f"  Answer preview: {result['answer'][:140]}...")


# ===========================================================================
# Main evaluation runner
# ===========================================================================
def run_evaluation() -> None:
    console.print(Panel.fit(
        "[bold]Golden Dataset Evaluation[/bold]\n"
        "5 QA pairs  ·  source precision + keyword coverage",
        border_style="blue",
    ))

    results_store  = []
    total_src_top1 = 0
    total_src_top4 = 0
    total_kw       = 0.0

    for i, qa in enumerate(GOLDEN_QA):
        console.print(f"\n[bold]Running {qa['id']} ({i+1}/5)...[/bold]")
        t0     = time.time()
        result = rag_query(qa["question"], top_k=4)
        elapsed = round(time.time() - t0, 1)

        scores = score_result(result, qa)
        results_store.append((qa, result, scores))

        total_src_top1 += int(scores["source_correct"])
        total_src_top4 += int(scores["any_source_hit"])
        total_kw       += scores["keyword_coverage"]

        print_detail(qa, result, scores)
        console.print(f"  [dim]{elapsed}s[/dim]")

    # ── Summary table ─────────────────────────────────────────────────────────
    console.print("\n")
    console.rule("[bold]Evaluation Report[/bold]")

    tbl = Table(show_lines=True, border_style="dim")
    tbl.add_column("ID",            style="bold cyan",  width=7)
    tbl.add_column("Source (top-1)", width=14)
    tbl.add_column("Source (top-4)", width=14)
    tbl.add_column("Keywords",       width=10)
    tbl.add_column("Top relevance",  width=13)
    tbl.add_column("Expected source",width=36)

    for qa, result, scores in results_store:
        tbl.add_row(
            qa["id"],
            "[green]✓[/green]" if scores["source_correct"] else "[red]✗[/red]",
            "[green]✓[/green]" if scores["any_source_hit"] else "[red]✗[/red]",
            f"{scores['keyword_coverage']}%",
            f"{scores['top_relevance']}%",
            qa["expected_source"][:36],
        )

    console.print(tbl)

    # ── Aggregate scores ──────────────────────────────────────────────────────
    console.print()
    console.print(f"  [bold]Source precision (top-1):[/bold]  {total_src_top1}/5  "
                  f"({round(total_src_top1/5*100)}%)")
    console.print(f"  [bold]Source hit rate  (top-4):[/bold]  {total_src_top4}/5  "
                  f"({round(total_src_top4/5*100)}%)")
    console.print(f"  [bold]Avg keyword coverage:    [/bold]  {round(total_kw/5)}%")

    # ── Interpretation ────────────────────────────────────────────────────────
    console.print()
    src_pct = round(total_src_top1 / 5 * 100)
    if src_pct == 100:
        console.print("  [bold green]Perfect source precision![/bold green] "
                      "Your retriever correctly routes every question.")
    elif src_pct >= 60:
        console.print("  [yellow]Decent precision.[/yellow] "
                      "Try increasing TOP_K or improving chunk overlap in ingest.py.")
    else:
        console.print("  [red]Low precision.[/red] "
                      "Consider a larger embedding model (e.g. all-mpnet-base-v2) "
                      "or smaller chunk windows.")

    console.print()


if __name__ == "__main__":
    run_evaluation()