import os
import string
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
import torch
import yaml

# =========================
# Load env
# =========================
load_dotenv()

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# =========================
# Providers
# =========================
EMBEDDING_PROVIDER = cfg["providers"]["embedding"]
RERANKER_PROVIDER  = cfg["providers"]["reranker"]
LLM_PROVIDER       = cfg["providers"]["llm"]

# =========================
# Local embedding config
# =========================
LOCAL_EMBEDDING_MODEL  = cfg["local"]["embedding_model"]
LOCAL_EMBEDDING_DEVICE = cfg["local"]["embedding_device"]


# =========================
# Local reranker config
# =========================
LOCAL_RERANKER_MODEL  = cfg["local"]["reranker_model"]
LOCAL_RERANKER_DEVICE = cfg["local"]["reranker_device"]

# =========================
# Models config for api / ollama
# =========================
EMBEDDING_MODEL = cfg["api"]["embedding_model"]
RERANKER_MODEL  = cfg["api"]["reranker_model"]
LLM_MODEL       = cfg["api"]["llm_model"]

# =========================
# Data
# =========================
DOC_PATH      = cfg["paths"]["docs"]
QUESTION_PATH = cfg["paths"]["questions"]

# =========================
# Chunking & Retrieval
# =========================
CHUNK_SIZE    = cfg["chunking"]["size"]
OVERLAP       = cfg["chunking"]["overlap"]

INITIAL_TOP_K = cfg["retrieval"]["initial_top_k"]
FINAL_TOP_K   = cfg["retrieval"]["final_top_k"]

# =========================
# LLM generation settings
# =========================

LLM_TEMPERATURE    = cfg["llm"]["temperature"]
LLM_TOP_P          = cfg["llm"]["top_p"]
LLM_MAX_TOKENS     = cfg["llm"]["max_tokens"]
LLM_REPEAT_PENALTY = cfg["llm"]["repeat_penalty"]


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
# cpu/gpu check
# =========================

def resolve_device(device_pref: str):
    if device_pref == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_pref


# =========================
# Embedding backend
# =========================


def init_embedder():
    if EMBEDDING_PROVIDER == "local":
        device = resolve_device(LOCAL_EMBEDDING_DEVICE)

        print(f"📦 Loading local embedding model:")
        print(f"   Model  : {LOCAL_EMBEDDING_MODEL}")
        print(f"   Device : {device}")

        return SentenceTransformer(
            LOCAL_EMBEDDING_MODEL,
            device=device,
            trust_remote_code=True
        )

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
        device = resolve_device(LOCAL_RERANKER_DEVICE)

        print(f"📦 Loading local reranker model:")
        print(f"   Model  : {LOCAL_RERANKER_MODEL}")
        print(f"   Device : {device}")

        return CrossEncoder(
            LOCAL_RERANKER_MODEL,
            device=device
        )

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
        inputs = [
            f"Query: {query}\nDocument: {documents[i]}" for i in candidate_idx]
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

# Answer the question using ONLY the context below.

# Context:
# {context}

# Rules:
# - Use only the context
# - If missing answer say: NOT IN CONTEXT

# Question:
# {question}

def build_prompt(context, question):
    return f"""

You are a question-answering assistant.

Rules:
1. Use ONLY the information in the provided context.
2. If the answer is not in the context, say: "I don't know based on the provided context."
3. Do NOT use prior knowledge.
4. Be concise and factual.

Context:
{context}

Question:
{question}

Answer:

""".strip()


def ask_llm(prompt):
    kwargs = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": LLM_TEMPERATURE,
        "top_p": LLM_TOP_P,
        "max_tokens": LLM_MAX_TOKENS,
    }

    # Ollama / local models only
    if LLM_PROVIDER == "ollama":
        kwargs["extra_body"] = {
            "repeat_penalty": LLM_REPEAT_PENALTY
        }

    resp = llm_client.chat.completions.create(**kwargs)
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


def embed_texts(texts, batch_size=8):
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
    print(
        f"Retrieval HIT@{FINAL_TOP_K}: {a['retrieval_hit']} / {a['total']} ({a['retrieval_hit']/a['total']:.1%})")

print(f"\nUnanswerable questions: {u['total']}")
print(f"Correct refusals: {u['correct_refusal']}")
print(f"Hallucinations: {u['hallucination']}")
