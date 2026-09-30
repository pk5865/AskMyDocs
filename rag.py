import os, re, time, uuid, socket, ipaddress, sqlite3, struct
from urllib.parse import urlparse
from pathlib import Path
import requests
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

STORE_DIR = Path(os.getenv("CHROMA_DIR", "chroma_store"))
STORE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = os.getenv("ASKMYDOCS_DB", str(STORE_DIR / "askmydocs.sqlite3"))

def _connect():
    return sqlite3.connect(DB_PATH, timeout=30)

with _connect() as _db:
    _db.execute("""CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL,
        name TEXT NOT NULL,
        chunk INTEGER NOT NULL,
        text TEXT NOT NULL,
        embedding BLOB NOT NULL
    )""")
    _db.execute("CREATE INDEX IF NOT EXISTS chunks_doc_id ON chunks(doc_id)")

def _pack_vector(values):
    return struct.pack(f"<{len(values)}f", *values)

def _unpack_vector(blob):
    return struct.unpack(f"<{len(blob) // 4}f", blob)

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
    vectors = embed(pieces, "RETRIEVAL_DOCUMENT")
    with _connect() as db:
        db.executemany("INSERT INTO chunks (id, doc_id, name, chunk, text, embedding) VALUES (?, ?, ?, ?, ?, ?)", [
            (f"{doc_id}-{n}", doc_id, name, n, piece, _pack_vector(vector))
            for n, (piece, vector) in enumerate(zip(pieces, vectors))
        ])
    return {"doc_id": doc_id, "name": name, "chunks": len(pieces)}

def list_docs():
    with _connect() as db:
        rows = db.execute("SELECT doc_id, name, COUNT(*) FROM chunks GROUP BY doc_id, name ORDER BY MIN(rowid)").fetchall()
    return [{"doc_id": doc_id, "name": name, "chunks": count} for doc_id, name, count in rows]

def delete_doc(doc_id):
    with _connect() as db:
        db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

# ---------- 4. Retrieve + 5. Generate ----------
def retrieve(question, k=4):
    with _connect() as db:
        rows = db.execute("SELECT text, name, chunk, embedding FROM chunks").fetchall()
    if not rows:
        return []
    query = embed([question], "RETRIEVAL_QUERY")[0]
    q_norm = sum(v * v for v in query) ** 0.5
    scored = []
    for text, name, chunk_no, packed in rows:
        vector = _unpack_vector(packed)
        denom = q_norm * (sum(v * v for v in vector) ** 0.5)
        score = sum(a * b for a, b in zip(query, vector)) / denom if denom else 0.0
        scored.append({"text": text, "source": name, "chunk": chunk_no, "score": round(score, 3)})
    return sorted(scored, key=lambda item: item["score"], reverse=True)[:k]

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
