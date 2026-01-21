## RAG Pipeline Overview

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
<pre>
.
├── data/
│ ├── docs.txt # Knowledge base
│ └── eval_questions.txt # Evaluation questions
├── rag_eval.py # Main RAG evaluation script
├── requirements.txt
└── README.md
</pre>
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
f

DOC_PATH = "data/docs.txt"
QUESTION_PATH = "data/eval_questions.txt"

docs.txt -> knowledge base
eval_question.txt -> question to test your rag


CHUNK_SIZE = 120
OVERLAP = 30
