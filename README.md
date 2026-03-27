# RAG Golden Dataset — Neural Networks & Deep Learning

A Retrieval-Augmented Generation system that ingests 4 YouTube video
transcripts, stores them in a local vector database, and evaluates 5
hand-crafted QA pairs to measure retrieval quality.

---

## Project structure

```
rag_assignment/
├──data
  ├──dataset.json
├── ingest.py          ← Step 1: fetch transcripts, embed, store
├── rag.py             ← Step 2: query function + smoke test
├── evaluate.py        ← Step 3: run all 5 golden QA pairs
├── requirements.txt   ← all Python dependencies
├── .env               ← your Anthropic API key (never commit this)
├── .gitignore
└── chroma_db/         ← created automatically by ingest.py
```

---

## Setup (do this once)

### 1. Create and activate virtual environment

**Mac / Linux**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
*(Takes 3–5 minutes. Downloads sentence-transformers and ChromaDB.)*

### 3. Add your Anthropic API key
Open `.env` and replace the placeholder:
```
ANTHROPIC_API_KEY=sk-ant-your-real-key-here
```
Get a key at https://console.anthropic.com/settings/keys

---

## Run order (always follow this order)

```bash
# 1. Ingest — run once, or re-run to refresh the database
python ingest.py

# 2. Smoke test — verify a single query works end to end
python rag.py

# 3. Full evaluation — runs all 5 golden QA pairs
python evaluate.py
```

---

## What each file does

| File | What it does |
|---|---|
| `ingest.py` | Pulls transcripts from YouTube, splits into 60-second chunks, embeds with `all-MiniLM-L6-v2`, saves to ChromaDB |
| `rag.py` | Embeds a question, retrieves top-4 chunks, sends to Claude with a strict system prompt |
| `evaluate.py` | Runs 5 golden QA pairs, scores source precision and keyword coverage, prints a summary table |

---

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError` | Run `source venv/bin/activate` first |
| `AuthenticationError` | Check your `.env` key — no quotes, no spaces |
| `Collection not found` | Run `python ingest.py` before `rag.py` |
| `TranscriptsDisabled` | Add `"hi-IN"` to the languages list in `ingest.py` |
| ChromaDB install fails on Windows | `pip install chromadb --only-binary :all:` |

---

## Videos ingested

1. 3Blue1Brown — *But what is a Neural Network?* (English)
2. 3Blue1Brown — *Transformers, the tech behind LLMs* (English)
3. CampusX — *What is Deep Learning?* (Hindi → auto-translated)
4. CodeWithHarry — *All About ML & Deep Learning* (Hindi → auto-translated)

# Results 
───────────────────────────────────────────────────── Evaluation Report ─────────────────────────────────────────────────────
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ ID      ┃ Source (top-1) ┃ Source (top-4) ┃ Keywords   ┃ Top relevance ┃ Expected source                      ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ QA-01   │ ✓              │ ✓              │ 80%        │ 86%           │ But what is a Neural Network?        │
├─────────┼────────────────┼────────────────┼────────────┼───────────────┼──────────────────────────────────────┤
│ QA-02   │ ✓              │ ✓              │ 100%       │ 68%           │ But what is a Neural Network?        │
├─────────┼────────────────┼────────────────┼────────────┼───────────────┼──────────────────────────────────────┤
│ QA-03   │ ✓              │ ✓              │ 100%       │ 81%           │ Transformers, the tech behind LLMs   │
├─────────┼────────────────┼────────────────┼────────────┼───────────────┼──────────────────────────────────────┤
│ QA-04   │ ✓              │ ✓              │ 80%        │ 79%           │ What is Deep Learning?               │
├─────────┼────────────────┼────────────────┼────────────┼───────────────┼──────────────────────────────────────┤
│ QA-05   │ ✓              │ ✓              │ 60%        │ 83%           │ All About ML & Deep Learning        │
└─────────┴────────────────┴────────────────┴────────────┴───────────────┴──────────────────────────────────────┘

  Source precision (top-1):  5/5  (100%)
  Source hit rate  (top-4):  5/5  (100%)
  Avg keyword coverage:      84%

  Perfect source precision! Your retriever correctly routes every question.
