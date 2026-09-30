import os, re, time, uuid, socket, ipaddress
from urllib.parse import urlparse
import requests, chromadb
from bs4 import BeautifulSoup
from pypdf import PdfReader
from google import genai
from google.genai import types

EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
NOT_FOUND = "I couldn't find this in the uploaded documents."

_client = None
def client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client

col = chromadb.PersistentClient(path=os.getenv("CHROMA_DIR", "chroma_store")).get_or_create_collection(
    "docs", metadata={"hnsw:space": "cosine"})

def embed(texts, task):
    """Turn text into vectors with Gemini. task = RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY."""
    out = []
    for i in range(0, len(texts), 50):
        r = client().models.embed_content(model=EMBED_MODEL, contents=texts[i:i + 50],
                                          config=types.EmbedContentConfig(task_type=task))
        out += [e.values for e in r.embeddings]
    return out

# ---------- 1. Extract text ----------
def pdf_text(stream):
    return "\n".join((p.extract_text() or "") for p in PdfReader(stream).pages)

def _check_url(url):
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError("Enter a valid http(s) URL")
    for info in socket.getaddrinfo(u.hostname, None):  # block internal addresses (SSRF)
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            raise ValueError("This URL is not allowed")

def web_text(url):
    _check_url(url)
    response = requests.get(url, timeout=10, headers={"User-Agent": "AskMyDocs"},
                            allow_redirects=False)
    if 300 <= response.status_code < 400:
        raise ValueError("Redirects are not allowed. Enter the destination URL directly")
    response.raise_for_status()
    html = response.text
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "footer"]):
        t.decompose()
    return soup.get_text(" ")

def youtube_text(url):
    from youtube_transcript_api import YouTubeTranscriptApi
    m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url)
    if not m:
        raise ValueError("Invalid YouTube URL")
    return " ".join(s.text for s in YouTubeTranscriptApi().fetch(m.group(1)))

# ---------- 2. Chunk ----------
def chunk(text, size=800, overlap=150):
    """Split into ~800-char pieces, ending at a sentence when possible, with overlap."""
    text = re.sub(r"\s+", " ", text).strip()
    out, i = [], 0
    while i < len(text):
        end = min(i + size, len(text))
        if end < len(text):
            cut = text.rfind(". ", i + size // 2, end)
            if cut != -1:
                end = cut + 1
        out.append(text[i:end].strip())
        if end >= len(text):
            break
        i = max(end - overlap, i + 1)
    return [c for c in out if len(c) > 30]

# ---------- 3. Store ----------
def add_document(name, text):
    pieces = chunk(text)
    if not pieces:
        raise ValueError("No readable text found in this source")
    doc_id = uuid.uuid4().hex[:10]
    col.add(ids=[f"{doc_id}-{n}" for n in range(len(pieces))], documents=pieces,
            embeddings=embed(pieces, "RETRIEVAL_DOCUMENT"),
            metadatas=[{"doc_id": doc_id, "name": name, "chunk": n} for n in range(len(pieces))])
    return {"doc_id": doc_id, "name": name, "chunks": len(pieces)}

def list_docs():
    docs = {}
    for m in col.get(include=["metadatas"])["metadatas"]:
        d = docs.setdefault(m["doc_id"], {"doc_id": m["doc_id"], "name": m["name"], "chunks": 0})
        d["chunks"] += 1
    return list(docs.values())

def delete_doc(doc_id):
    col.delete(where={"doc_id": doc_id})

# ---------- 4. Retrieve + 5. Generate ----------
def retrieve(question, k=4):
    n = col.count()
    if n == 0:
        return []
    r = col.query(query_embeddings=embed([question], "RETRIEVAL_QUERY"), n_results=min(k, n),
                  include=["documents", "metadatas", "distances"])
    return [{"text": t, "source": m["name"], "chunk": m["chunk"], "score": round(1 - d, 3)}
            for t, m, d in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]

def generate(prompt):
    """Call Gemini. Retry when Google is busy (503/429), then try FALLBACK_MODELS from .env."""
    models = [CHAT_MODEL] + [m.strip() for m in os.getenv("FALLBACK_MODELS", "").split(",") if m.strip()]
    last = None
    for model in models:
        for attempt in range(4):
            try:
                return client().models.generate_content(model=model, contents=prompt).text or ""
            except Exception as e:
                last = e
                if any(k in str(e) for k in ("503", "429", "UNAVAILABLE")):
                    time.sleep(2 ** attempt)  # wait 1s, 2s, 4s, 8s
                else:
                    raise
    raise last

def answer(question):
    hits = retrieve(question)
    if not hits:
        return {"answer": NOT_FOUND, "sources": []}
    ctx = "\n\n".join(f"[{i + 1}] ({h['source']}) {h['text']}" for i, h in enumerate(hits))
    prompt = (f"Answer the question using ONLY the context below. Cite sources like [1] or [2]. "
              f"If the context does not contain the answer, reply exactly: {NOT_FOUND}\n\n"
              f"Context:\n{ctx}\n\nQuestion: {question}")
    return {"answer": generate(prompt).strip(), "sources": hits}

# ---------- 6. Evaluate ----------
def evaluate(cases):
    """cases = [{'q': question, 'expect': keyword that must appear}]. Measures retrieval and answer accuracy."""
    rows = []
    for c in cases:
        res, exp = answer(c["q"]), c["expect"].lower()
        rows.append({"q": c["q"], "expect": c["expect"],
                     "retrieval_hit": any(exp in h["text"].lower() for h in res["sources"]),
                     "answer_ok": exp in res["answer"].lower(), "answer": res["answer"]})
    n = max(len(rows), 1)
    return {"rows": rows, "retrieval_rate": round(100 * sum(r["retrieval_hit"] for r in rows) / n),
            "answer_rate": round(100 * sum(r["answer_ok"] for r in rows) / n)}
