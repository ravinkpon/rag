import os
import numpy as np
import string
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from tqdm import tqdm  # Recommended for progress bars

# =========================
# Config
# =========================
DOC_PATH = "data/docs.txt"
QUESTION_PATH = "data/eval_questions.txt"

CHUNK_SIZE = 120
OVERLAP = 30

INITIAL_TOP_K = 15   # Fetch more candidates for the reranker
FINAL_TOP_K = 3      # Give the LLM only the best 3

EMBEDDING_MODEL = "google/embeddinggemma-300m"
# "all-MiniLM-L6-v2" #"Qwen/Qwen3-Embedding-0.6B",  "BAAI/bge-small-en-v1.5"
# 
# Excellent choice for CPU. Fast and accurate enough for reranking.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2" 
# "cross-encoder/ms-marco-MiniLM-L-12-v2" Qwen/Qwen3-Reranker-0.6B "tomaarsen/Qwen3-Reranker-0.6B-seq-cls" "BAAI/bge-reranker-base" (not works in 6core cpu)
# "cross-encoder/ms-marco-MiniLM-L-6-v2" (works in cpu)
LLM_MODEL = "gpt-4o-mini"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# =========================
# Models
# =========================
print("Loading models...")
# FIX 1: Qwen often requires trust_remote_code=True
embedder = SentenceTransformer(EMBEDDING_MODEL,trust_remote_code=True ) #
reranker = CrossEncoder(RERANKER_MODEL)

# =========================
# Utilities
# =========================
def chunk_text(text, chunk_size, overlap):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += chunk_size - overlap
    return chunks

def retrieve_candidates(query_emb, doc_embs, k):
    scores = np.dot(doc_embs, query_emb)
    return np.argsort(scores)[-k:][::-1]

def rerank(query, documents, candidate_idx, top_k):
    pairs = [[query, documents[i]] for i in candidate_idx]
    scores = reranker.predict(pairs)
    best_indices = np.argsort(scores)[-top_k:][::-1]
    return [candidate_idx[i] for i in best_indices], [scores[i] for i in best_indices]

def build_prompt(context, question):
    return f"""
Answer the question using ONLY the context below.

Context:
{context}

Rules:
- You must answer strictly based on the provided context.
- If the context does not contain the answer, respond with exactly: NOT IN CONTEXT
- Do not use outside knowledge.

Question:
{question}
""".strip()

def ask_llm(prompt):
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# =========================
# Load data
# =========================
# Ensure files exist to avoid crash
if not os.path.exists(DOC_PATH):
    # Create dummy data if file missing (for testing)
    print(f"Warning: {DOC_PATH} not found. Creating dummy data.")
    os.makedirs("data", exist_ok=True)
    with open(DOC_PATH, "w", encoding="utf-8") as f:
        f.write("Qwen is a large language model series. RAG stands for Retrieval Augmented Generation. " * 100)
    with open(QUESTION_PATH, "w", encoding="utf-8") as f:
        f.write("[A] What does RAG stand for?\n[U] Who is the president of Mars?")

with open(DOC_PATH, encoding="utf-8") as f:
    raw_text = f.read()

documents = chunk_text(raw_text, CHUNK_SIZE, OVERLAP)
print(f"Loaded {len(documents)} chunks")

with open(QUESTION_PATH, encoding="utf-8") as f:
    questions = [q.strip() for q in f if q.strip()]

# =========================
# Precompute embeddings
# =========================
print("Embedding documents (this may take a moment on CPU)...")
# FIX 2: Add progress bar for clarity
doc_embeddings = embedder.encode(documents, normalize_embeddings=True, show_progress_bar=True)

# =========================
# Evaluation state
# =========================
results = {
    "answerable": dict(total=0, correct=0, fail=0, retrieval_hit=0),
    "unanswerable": dict(total=0, correct_refusal=0, hallucination=0),
}

# =========================
# Main evaluation loop
# =========================
for raw_q in tqdm(questions, desc="Processing Questions"): # Add progress bar
    
    is_answerable = raw_q.startswith("[A]")
    question = raw_q[3:].strip()

    if is_answerable:
        results["answerable"]["total"] += 1
    else:
        results["unanswerable"]["total"] += 1

    # --- 1. Retrieval (Bi-Encoder) ---
    # FIX 3: ADD INSTRUCTION PREFIX FOR QWEN
    # Qwen-Embedding is asymmetric; it needs to know this is a "Query" looking for an answer.
    instruction = "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "
    q_emb = embedder.encode([instruction + question], normalize_embeddings=True)[0]
    
    candidate_idx = retrieve_candidates(q_emb, doc_embeddings, INITIAL_TOP_K)

    # --- 2. Reranking (Cross-Encoder) ---
    top_idx, rerank_scores = rerank(
        question, documents, candidate_idx, FINAL_TOP_K
    )

    context = "\n".join(documents[i] for i in top_idx)
    retrieved_text_lower = " ".join(documents[i].lower() for i in top_idx)

    # --- 3. Generation ---
    prompt = build_prompt(context, question)
    answer = ask_llm(prompt)
    
    # Use tqdm.write so it doesn't break the progress bar
    tqdm.write("\n" + "="*50)
    tqdm.write(f"Q: {question}")
    tqdm.write(f"A: {answer}")
    tqdm.write("-"*50 + "\n")
    tqdm.write(f"Retrived: {top_idx}")
    tqdm.write("="*50 + "\n")
    answer_lc = answer.lower()

    # --- 4. Evaluation ---
    if is_answerable:
        if "not in context" in answer_lc:
            results["answerable"]["fail"] += 1
        else:
            results["answerable"]["correct"] += 1
            clean_q = question.lower().translate(str.maketrans('', '', string.punctuation))
            keywords = [w for w in clean_q.split() if len(w) > 3]
            if any(k in retrieved_text_lower for k in keywords):
                results["answerable"]["retrieval_hit"] += 1
    else:
        if "not in context" in answer_lc:
            results["unanswerable"]["correct_refusal"] += 1
        else:
            results["unanswerable"]["hallucination"] += 1

# =========================
# Summary
# =========================
print("\n" + "=" * 40)
print("EVALUATION SUMMARY")
a = results["answerable"]
u = results["unanswerable"]

print(f"\nAnswerable questions: {a['total']}")
print(f"Correct answers: {a['correct']}")
print(f"Failed answers: {a['fail']}")
if a['total'] > 0:
    print(f"Retrieval HIT@{FINAL_TOP_K}: {a['retrieval_hit']} / {a['total']} ({a['retrieval_hit']/a['total']:.1%})")

print(f"\nUnanswerable questions: {u['total']}")
print(f"Correct refusals: {u['correct_refusal']}")
print(f"Hallucinations: {u['hallucination']}")