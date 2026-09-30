# AskMyDocs

Ask questions about uploaded PDF/TXT files, webpages, and YouTube transcripts. The app extracts and chunks text, stores Gemini embeddings and source text in SQLite, retrieves relevant passages, then asks Gemini to answer from those passages with source citations.

**Stack:** Flask, SQLite, Gemini API, vanilla JavaScript.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Add your Gemini API key to .env
python app.py
```

Open `http://localhost:5000`. The API key is required for embedding documents and generating answers. Keep `.env` private; only `.env.example` belongs in Git.

## Deploy on Render

This repository includes `render.yaml` for a free Render web service. Create a Blueprint from the GitHub repository, then add `GEMINI_API_KEY` under the service's Environment settings. Do not commit the key.

The free service sleeps after inactivity and its filesystem is temporary. SQLite's uploaded document index can be lost after a restart or redeploy. This app uses one shared document collection without user accounts, so do not upload confidential documents to a public deployment. Anyone who can reach the app can use its Gemini-backed endpoints and consume the configured API quota.

## Limits

- Documents are shared by every visitor; there are no user accounts or per-user collections.
- Uploaded files are limited to 10 MB.
- PDFs need an extractable text layer; scanned image PDFs do not have OCR.
- Evaluation uses simple keyword checks and is not a formal quality benchmark.
