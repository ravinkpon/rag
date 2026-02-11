from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import JinaRerank
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from pydantic import SecretStr
from pathlib import Path
import os

load_dotenv()

# ---------------- PATHS ----------------
pdf_path = Path(r"D:\practice\rag\rag_pdf\data\pdf\DSML.pdf")
faiss_cache_path = Path(r"D:\practice\rag\rag_pdf\data\faiss_data")

# ---------------- API KEY CHECK ----------------
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not set")
if not os.getenv("CLAUDE_API_KEY"):
    raise RuntimeError("CLAUDE_API_KEY not set")
if not os.getenv("JINA_API_KEY"):
    raise RuntimeError("JINA_API_KEY not set")
# ---------------- EMBEDDINGS ----------------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key= SecretStr(os.environ["OPENAI_API_KEY"]),
)

# ---------------- FAISS CACHE CHECK ----------------
if faiss_cache_path.exists():
    print("FAISS index found. Loading from disk (skipping embedding)...")

    db = FAISS.load_local(
        faiss_cache_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("Total vectors loaded:", db.index.ntotal)

else:
    print("FAISS index not found. Creating embeddings...")

    # -------- Load PDF --------
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    print(f"Total pages: {len(docs)}")

    # -------- Chunk --------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        add_start_index=True
    )

    textchunk = splitter.split_documents(docs)
    print(f"Total chunks: {len(textchunk)}")

    # -------- Embed + FAISS --------
    db = FAISS.from_documents(textchunk, embeddings)
    print("Total vectors stored:", db.index.ntotal)

    # -------- Save FAISS --------
    faiss_cache_path.mkdir(parents=True, exist_ok=True)
    db.save_local(faiss_cache_path)
    print("FAISS index saved successfully")

    # -------- ask the question --------
query = input("please ask your question \n")

    # -------- base top 8 retriever  --------
    
# docs = db.similarity_search(query, k=3) 

retriever = db.as_retriever(
    search_type ="similarity",
    search_kwargs={"k": 8}
)

    #------ Create BM25 Retriever--------

r_docs = retriever.invoke(query) 

print(f"Retrieved {len(r_docs)} documents\n")

for i, doc in enumerate(r_docs, 1):
    print(f"\n--- Chunk {i} ---")
    print(doc.page_content[:300])

    # -------- JINA Reranker --------

compressor = JinaRerank(
    model = "jina-reranker-v1-base-en",
    top_n=3
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever ,
)

import time

start = time.time()
docs = compression_retriever.invoke(query)
end = time.time()

print("Rerank time:", end - start)

print(f"top 3 jina retriever {len(docs)}")

# print(f"top 3 jina retriever {docs}")

# for i, doc in enumerate(docs, 1):
#     print(f"\nRank {i}")
#     print("Score:", doc.metadata.get("relevance_score"))
#     print("Page:", doc.metadata.get("page"))
#     print(doc.page_content[:200])


#     # -------- Context for llm --------
def build_context(docs):
    context = ""
    for i, doc in enumerate(docs, 1):
        context += f"\n[Source {i} | Page {doc.metadata.get('page')}]\n"
        context += doc.page_content.strip() + "\n"
    return context

SYSTEM_PROMPT = """You are a helpful assistant.
Answer ONLY using the provided context.
If the answer is not in the context, say "I don't know".
"""

def build_prompt(question, context):
    return f"""{SYSTEM_PROMPT}

Context:
{context}

Question:
{question}

Answer:
"""

llm = ChatAnthropic(
    model_name="claude-3-haiku-20240307",
    temperature=0,
    api_key= SecretStr(os.environ["CLAUDE_API_KEY"]),
    timeout= 60,
    stop=None,
)

context = build_context(docs)
prompt = build_prompt(query, context)

response = llm.invoke(prompt)

print(response.content)

print("\nSources:")
for i, doc in enumerate(docs, 1):
    print(f"{i}. Page {doc.metadata.get('page')}")


