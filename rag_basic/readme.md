## RAG Pipeline with Evaluation

<pre>
Documents  ──▶  Chunks  ──▶  Embeddings  ──▶  Vector Search  ──▶  Reranker
                                                         │
Question   ──▶  Query Embedding  ─────────────────────────┘
                                                         │
                                                     Top-K Context
                                                         │
                                                      LLM Answer
                                                         │
                                                      Evaluation
</pre>
---
## Project Structure

```\
.
├── data/
│ ├── docs.txt # Knowledge base
│ └── eval_questions.txt # Evaluation questions
├── rag_eval.py # Main RAG evaluation script
├── requirements.txt
└── README.md
└── .env # create the file and follow instruction below
```
---
## 📝 Question Format

**Questions are labeled to enable automatic evaluation:**

<pre>
[A] What does RAG stand for?
[U] Who is the president of Mars?

[A] → Answer exists in documents
[U] → Answer does NOT exist (model must refuse)
</pre>
---
## Models Used

| Component | Model                                  | Purpose            |
| --------- | -------------------------------------- | ------------------ |
| Embedding | `google/embeddinggemma-300m`           | Semantic retrieval |
| Reranker  | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Precise ranking    |
| LLM       | `gpt-4o-mini`                          | Answer generation  |

you can choose your own model 
---

## ⚙️ Installation

Make sure Python is 3.12.8 
```bash
pip install -r requirements.txt
```

## Set API Key 

Create .env file in the Project root 
paste this and change with you key
```
OPENAI_API_KEY="sk-proj-.....................wOUA"
MODELSCOPE_API_KEY="ms-6a..................5b"
OLLAMA_BASE_URL=http://localhost:11434/v1
```
---

## ⚙️ Configuration File

All models, providers, and generation settings are controlled through a single `config.yaml` file.\
The code reads settings from `config.yaml` at runtime \
This allows you to:

* switch between local and cloud models
* change embedding, reranker, or LLM models
* tune chunking, retrieval, and generation parameters
* experiment without modifying the Python code


---


## ▶️ Run the Project

```bash
python main.py
```

The script will:
- Embed documents
- Retrieve and rerank context
- Generate answers
- Print evaluation results

---

## 📊 Evaluation Metrics

### Answerable Questions
- Correct answers
- Failed answers
- Retrieval HIT@K

### Unanswerable Questions
- Correct refusals
- Hallucinations

---

## 🎯 Why This Project Matters

This project demonstrates:
- Proper document chunking
- Two-stage retrieval (bi-encoder + cross-encoder)
- Context-restricted generation
- Hallucination detection
- Practical RAG evaluation logic

---