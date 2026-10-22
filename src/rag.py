# LIBRARIES
import os
import re
import chromadb
from chromadb.utils import embedding_functions
import ollama
############################################################################################################

# SET PATHS AND SETTINGS
# the vector database created by data_ingestion.py (in the project root)
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
# must be the same collection name used in data_ingestion.py
COLLECTION_NAME = "australian_visa_docs"
# the local language model that writes the answers (runs through Ollama)
OLLAMA_MODEL = "llama3.1:8b"

# instructions sent to the model with every question
SYSTEM_PROMPT = """You are a helpful assistant for international students asking \
about Australian student visa rules. Answer using ONLY the provided context. \
If the context contains relevant information, answer directly and precisely — \
quote exact numbers, hours, or conditions exactly as stated rather than \
paraphrasing them loosely. If the context describes an exception that applies \
only to a specific group (e.g. a particular degree type), clearly say it is an \
exception and state what the general rule is for everyone else, if the general \
rule is also present in the context. Only say "I don't have enough information \
in the provided documents to answer that" if the context truly contains nothing \
relevant to the question. Cite the source document and page number for every \
claim, like this: (Source: filename.pdf, p.3)."""

############################################################################################################
# FUNCTIONS DECLARATION
############################################################################################################

############################################################################################################
# ChromaDB
# open the saved vector database and return the collection of chunks
def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    # must be the same embedding model that was used in data_ingestion.py
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-en-v1.5"
    )
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

############################################################################################################
# Retrieval
# find the k chunks that are most likely to contain the answer to the question
def retrieve(query, k=6):
    collection = get_collection()
    # get a bigger pool of chunks first (at least 20) so there is more to choose from
    fetch_k = max(k * 4, 20)
    results = collection.query(query_texts=[query], n_results=fetch_k)
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # very common words that are ignored when matching keywords
    STOPWORDS = {
        "the", "and", "for", "you", "your", "with", "this", "that",
        "visa", "student", "students", "apply", "application", "can",
        "while", "have", "must", "are", "will", "not", "from", "when",
    }

    # keywords in the question (words with 3+ letters, without the stopwords)
    query_words = set(re.findall(r"[a-z]{3,}", query.lower())) - STOPWORDS

    # keyword re-ranking: every shared keyword lowers the distance by 0.3
    # a lower score means a better chunk
    scored = []
    for chunk, meta, dist in zip(chunks, metadatas, distances):
        chunk_words = set(re.findall(r"[a-z]{3,}", chunk.lower())) - STOPWORDS
        # number of keywords the question and the chunk share
        overlap = len(query_words & chunk_words)
        adjusted_score = dist - (overlap * 0.3)
        scored.append((adjusted_score, chunk, meta, dist))

    # sort by the new score and keep the best k chunks
    scored.sort(key=lambda x: x[0])
    top = scored[:k]
    return [(chunk, meta, dist) for _, chunk, meta, dist in top]

############################################################################################################
# Generation
# join the chunks into one text, each labelled with its file name and page number
def build_context(retrieved):
    parts = []
    for chunk, meta, _ in retrieved:
        parts.append(f"[{meta['source']}, p.{meta['page']}]\n{chunk}")
    return "\n\n---\n\n".join(parts)

# send the chunks and the question to the language model and return its answer
def generate_answer(query, retrieved, model=OLLAMA_MODEL):
    context = build_context(retrieved)
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"]

############################################################################################################
# Full pipeline
# retrieve the chunks -> generate the answer -> return the answer with its sources
# this is the function used by app.py
def rag_answer(query, k=6, model=OLLAMA_MODEL):
    retrieved = retrieve(query, k=k)
    answer = generate_answer(query, retrieved, model=model)
    # sources shown in the app (file name, page number, score)
    sources = [
        {"source": m["source"], "page": m["page"], "score": round(1 - d, 3)}
        for _, m, d in retrieved
    ]
    return {"answer": answer, "sources": sources, "context_chunks": retrieved}