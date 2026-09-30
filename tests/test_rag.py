import os, sys, tempfile, hashlib
os.environ["CHROMA_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import rag

def fake_embed(texts, task):  # stand-in for Gemini so tests run offline
    out = []
    for t in texts:
        v = [0.0] * 256
        for w in t.lower().split():
            v[int(hashlib.md5(w.strip(".,?").encode()).hexdigest(), 16) % 256] += 1
        out.append(v)
    return out
rag.embed = fake_embed

class R:  # fake Gemini reply
    text = "Refunds take 30 days [1]"
class M:
    def generate_content(self, **k): return R()
class C: models = M()
rag.client = lambda: C()

def test_chunk_overlap():
    c = rag.chunk("Sentence number one is here. " * 100)
    assert len(c) > 2 and all(len(x) <= 800 for x in c)

def test_pipeline():
    a = rag.add_document("policy.txt", "Our refund policy allows refunds within 30 days of purchase. " * 3)
    assert len(rag.list_docs()) == 1
    top = rag.retrieve("how many days for refunds?", k=1)[0]
    assert top["source"] == "policy.txt"
    res = rag.answer("refund days?")
    assert "30" in res["answer"] and res["sources"]
    ev = rag.evaluate([{"q": "refund days?", "expect": "30 days"}])
    assert ev["retrieval_rate"] == 100 and ev["answer_rate"] == 100
    rag.add_document("other.txt", "Bananas grow in tropical climates and are rich in potassium. " * 3)
    assert [d["name"] for d in rag.list_docs()] == ["other.txt"]
    assert rag.retrieve("where do bananas grow?", k=1)[0]["source"] == "other.txt"
    rag.delete_doc(rag.list_docs()[0]["doc_id"])
    assert rag.list_docs() == []

def test_ssrf_blocked():
    try:
        rag.web_text("http://127.0.0.1:8000/x"); assert False
    except ValueError:
        pass

def test_web_redirects_are_rejected(monkeypatch):
    requested = {}
    class Response:
        status_code = 302
        text = ""
    def fake_get(url, **kwargs):
        requested.update(kwargs)
        return Response()
    monkeypatch.setattr(rag.socket, "getaddrinfo", lambda *_: [
        (rag.socket.AF_INET, rag.socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
    ])
    monkeypatch.setattr(rag.requests, "get", fake_get)
    try:
        rag.web_text("https://example.com")
        assert False, "redirects should be rejected"
    except ValueError as exc:
        assert "Redirects are not allowed" in str(exc)
    assert requested["allow_redirects"] is False
