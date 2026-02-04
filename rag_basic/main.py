import os
import string
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI

# =========================
# Load env
# =========================
load_dotenv()

# =========================
# Providers
# =========================
EMBEDDING_PROVIDER = "modelscope"   # local | openai | modelscope | ollama
RERANKER_PROVIDER  = "modelscope"        # local | modelscope
LLM_PROVIDER       = "openai"       # ollama | openai | modelscope

# =========================
# Models
# =========================
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
RERANKER_MODEL  = "Qwen/Qwen3-Reranker-8B"#"cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL       = "gpt-4o-mini"   # ollama model OR gpt-4o-mini etc

# =========================
# Data
# =========================
DOC_PATH = "data/docs.txt"
QUESTION_PATH = "data/eval_questions.txt"

CHUNK_SIZE = 120
OVERLAP = 30
INITIAL_TOP_K = 15
FINAL_TOP_K = 3

# =========================
# LLM Client
# =========================
if LLM_PROVIDER == "ollama":
    llm_client = OpenAI(
        base_url=os.getenv("OLLAMA_BASE_URL"),
        api_key="ollama"
    )

elif LLM_PROVIDER == "openai":
    llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

elif LLM_PROVIDER == "modelscope":
    llm_client = OpenAI(
        base_url="https://api-inference.modelscope.ai/v1",
        api_key=os.getenv("MODELSCOPE_API_KEY")
    )

else:
    raise ValueError("Invalid LLM_PROVIDER")

print(f"✅ Using LLM provider: {LLM_PROVIDER}")
print(f"Model        : {LLM_MODEL}")

# =========================
# Embedding backend
# =========================
def init_embedder():
    if EMBEDDING_PROVIDER == "local":
        return SentenceTransformer("google/embeddinggemma-300m", trust_remote_code=True)

    elif EMBEDDING_PROVIDER == "openai":
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    elif EMBEDDING_PROVIDER == "modelscope":
        return OpenAI(
            base_url="https://api-inference.modelscope.ai/v1",
            api_key=os.getenv("MODELSCOPE_API_KEY")
        )

    elif EMBEDDING_PROVIDER == "ollama":
        return OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            api_key="ollama"
        )

    else:
        raise ValueError("Invalid EMBEDDING_PROVIDER")

embedder = init_embedder()

def embed_texts(texts):
    if EMBEDDING_PROVIDER == "local":
        return embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    resp = embedder.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        encoding_format="float"
    )
    return np.array([d.embedding for d in resp.data])

# =========================
# Reranker backend
# =========================
def init_reranker():
    if RERANKER_PROVIDER == "local":
        return CrossEncoder(RERANKER_MODEL)

    elif RERANKER_PROVIDER == "modelscope":
        return OpenAI(
            base_url="https://api-inference.modelscope.ai/v1",
            api_key=os.getenv("MODELSCOPE_API_KEY")
        )

    else:
        raise ValueError("Invalid RERANKER_PROVIDER")

reranker = init_reranker()

def rerank(query, documents, candidate_idx, top_k):
    if RERANKER_PROVIDER == "local":
        pairs = [[query, documents[i]] for i in candidate_idx]
        scores = reranker.predict(pairs)

    else:
        inputs = [f"Query: {query}\nDocument: {documents[i]}" for i in candidate_idx]
        resp = reranker.embeddings.create(model=RERANKER_MODEL, input=inputs)
        scores = [np.mean(d.embedding) for d in resp.data]

    best = np.argsort(scores)[-top_k:][::-1]
    return [candidate_idx[i] for i in best]

# =========================
# Utilities
# =========================
def chunk_text(text, chunk_size, overlap):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size]))
        i += chunk_size - overlap
    return chunks

def retrieve_candidates(query_emb, doc_embs, k):
    scores = np.dot(doc_embs, query_emb)
    return np.argsort(scores)[-k:][::-1]

def build_prompt(context, question):
    return f"""
Answer the question using ONLY the context below.

Context:
{context}

Rules:
- Use only the context
- If missing answer say: NOT IN CONTEXT

Question:
{question}
""".strip()

def ask_llm(prompt):
    resp = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300,
    )
    return resp.choices[0].message.content

# =========================
# Load data
# =========================
if not os.path.exists(DOC_PATH):
    os.makedirs("data", exist_ok=True)
    with open(DOC_PATH, "w") as f:
        f.write("RAG means Retrieval Augmented Generation. " * 200)
    with open(QUESTION_PATH, "w") as f:
        f.write("[A] What does RAG stand for?\n[U] Who is president of Mars?")

with open(DOC_PATH, encoding="utf-8") as f:
    raw_text = f.read()

documents = chunk_text(raw_text, CHUNK_SIZE, OVERLAP)

with open(QUESTION_PATH) as f:
    questions = [q.strip() for q in f if q.strip()]

# =========================
# Embed docs
# =========================
print("Embedding documents...")
def embed_texts(texts, batch_size=16):
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i+batch_size]

        if EMBEDDING_PROVIDER == "local":
            embs = embedder.encode(batch, normalize_embeddings=True)
        else:
            resp = embedder.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
                encoding_format="float"
            )
            embs = [d.embedding for d in resp.data]

        all_embeddings.extend(embs)

    return np.array(all_embeddings)

doc_embeddings = embed_texts(documents, batch_size=8)


# =========================
# Eval loop
# =========================

results = {
    "answerable": dict(total=0, correct=0, fail=0, retrieval_hit=0),
    "unanswerable": dict(total=0, correct_refusal=0, hallucination=0),
}



for raw_q in tqdm(questions, desc="Processing Questions"):
    is_answerable = raw_q.startswith("[A]")
    question = raw_q[3:].strip()

    if is_answerable:
        results["answerable"]["total"] += 1
    else:
        results["unanswerable"]["total"] += 1

    # ---- Retrieval ----
    q_emb = embed_texts([question])[0]
    candidate_idx = retrieve_candidates(q_emb, doc_embeddings, INITIAL_TOP_K)
    top_idx = rerank(question, documents, candidate_idx, FINAL_TOP_K)

    context = "\n".join(documents[i] for i in top_idx)
    retrieved_text_lower = " ".join(documents[i].lower() for i in top_idx)

    # ---- Generation ----
    prompt = build_prompt(context, question)
    answer = ask_llm(prompt)

    tqdm.write("="*60)
    tqdm.write(f"Q: {question}")
    tqdm.write(f"A: {answer}")
    tqdm.write(f"Retrieved chunks: {top_idx}")
    tqdm.write("="*60)

    answer_lc = answer.lower()

    # ---- Evaluation ----
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
