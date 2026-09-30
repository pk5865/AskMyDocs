import os
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import rag

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

def fail(e, code=400):
    return jsonify({"error": str(e)}), code

@app.get("/")
def home():
    return send_from_directory("static", "index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/docs")
def docs():
    return jsonify(rag.list_docs())

@app.delete("/api/docs/<doc_id>")
def remove(doc_id):
    rag.delete_doc(doc_id)
    return {"ok": True}

@app.post("/api/upload")
def upload():
    try:
        f, url = request.files.get("file"), (request.form.get("url") or "").strip()
        if f and f.filename:
            name = secure_filename(f.filename)
            ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
            if ext not in ("pdf", "txt"):
                return fail("Only PDF or TXT files are allowed")
            text = rag.pdf_text(f.stream) if ext == "pdf" else f.read().decode("utf-8", "ignore")
        elif url:
            name = url
            text = rag.youtube_text(url) if ("youtube.com" in url or "youtu.be" in url) else rag.web_text(url)
        else:
            return fail("Send a file or a URL")
        return jsonify(rag.add_document(name, text))
    except ValueError as e:
        return fail(e)
    except Exception as e:
        return fail(f"Could not process this source: {e}", 500)

@app.post("/api/ask")
def ask():
    q = (request.get_json(silent=True) or {}).get("question", "").strip()
    if not q:
        return fail("Type a question")
    try:
        return jsonify(rag.answer(q))
    except Exception as e:
        return fail(f"Answer failed: {e}", 500)

@app.post("/api/evaluate")
def evaluate():
    cases = (request.get_json(silent=True) or {}).get("cases", [])
    cases = [c for c in cases if c.get("q") and c.get("expect")][:20]
    if not cases:
        return fail("Add at least one test: question | expected keyword")
    try:
        return jsonify(rag.evaluate(cases))
    except Exception as e:
        return fail(f"Evaluation failed: {e}", 500)

if __name__ == "__main__":
    app.run(debug=False, port=int(os.getenv("PORT", 5000)))
