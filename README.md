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

  QA-01  ✓ CORRECT  Keywords: 80%  Top relevance: 86%
  Q: What is the difference between a weight and a bias in a neural network, and why ...
  Retrieved: But what is a Neural Network?
  Missing keywords: ['trainable']
  Answer preview: A weight is the multiplier attached to each connection between neurons, determining how strongly one       
neuron’s output influences the next — i...
  53.5s

Running QA-02 (2/5)...

  QA-02  ✓ CORRECT  Keywords: 100%  Top relevance: 68%
  Q: What role do hidden layers play when recognising handwritten digits, according t...
  Retrieved: But what is a Neural Network?
  Answer preview: Hidden layers sit between the input and output of the network and are responsible for learning
progressively more abstract visual features o...
  1.0s

Running QA-03 (3/5)...

  QA-03  ✓ CORRECT  Keywords: 100%  Top relevance: 81%
  Q: Why is softmax applied at the final step of a transformer, and what are logits?...
  Retrieved: Transformers, the tech behind LLMs
  Answer preview: Logits are the raw real‑valued scores produced at the output of a transformer: the final hidden state for  
the last position is multiplied by...
  1.2s

Running QA-04 (4/5)...

  QA-04  ✓ CORRECT  Keywords: 80%  Top relevance: 79%
  Q: What core limitation of traditional machine learning does deep learning address,...
  Retrieved: What is Deep Learning?
  Missing keywords: ['manual']
  Answer preview: Deep learning overcomes the **feature‑engineering bottleneck** of traditional machine learning by
automatically learning useful representati...
  1.9s

Running QA-05 (5/5)...

  QA-05  ✓ CORRECT  Keywords: 60%  Top relevance: 83%
  Q: What is overfitting in machine learning, how does it arise in a supervised learn...
  Retrieved: All About ML & Deep Learning
  Missing keywords: ['generalise', 'test']
  Answer preview: Overfitting is when a model memorises the training data—including its noise, outliers, and random
fluctuations—rather than learning the true...
  1.3s
<img width="1778" height="512" alt="image" src="https://github.com/user-attachments/assets/ed3d951f-1532-4ae4-9ecb-9bf55b166c71" />

  Source precision (top-1):  5/5  (100%)
  Source hit rate  (top-4):  5/5  (100%)
  Avg keyword coverage:      84%

  Perfect source precision! Your retriever correctly routes every question.
