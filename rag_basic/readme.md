## RAG Pipeline with Evaluation

<pre>
Documents  ──▶  Chunks  ──▶  Embeddings  ──▶  Vector Search  ──▶  Reranker
                                                         │
Question   ──▶  Query Embedding  ─────────────────────────┘
                                                         │
                                                     Top-3 Context
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

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

### Set OpenAI API Key

**Windows (PowerShell)**
```powershell
setx OPENAI_API_KEY "your_api_key_here"
```

**Linux / macOS**
```bash
export OPENAI_API_KEY="your_api_key_here"
```

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