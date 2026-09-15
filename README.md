# PaperLens — Research Paper RAG Agent

A local FastAPI application with a traditional browser frontend. It indexes one uploaded research paper and answers only from retrieved source excerpts. Gemini is called from the server, so the browser never sees the API key.

## Run it

1. Create a Gemini API key at [Google AI Studio](https://aistudio.google.com/app/apikey).
2. In the project directory, create a virtual environment and install the dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env`, paste the API key after `GEMINI_API_KEY=`, then run:

   ```bash
   python server.py
   ```

Open `http://127.0.0.1:8501` in your browser. That one command starts both the API and the chat frontend.

## What happens after upload

1. Text is extracted page-by-page from PDFs (or from DOCX/TXT).
2. It is split into small, overlapping excerpts with page/location metadata.
3. Gemini creates document embeddings; the local in-memory vector index retrieves the five most relevant excerpts for each question.
4. Gemini answers using only those excerpts, at low temperature, with inline `[S1]` source labels.
5. The chat UI displays matching excerpts and PDF page numbers below every answer.

The index is discarded when a different document is uploaded or the app session ends. Optional: install `faiss-cpu` to use FAISS automatically; without it the built-in NumPy cosine index is used.

## Demo checklist

- Upload a text-based PDF, ask a factual question, and expand **Retrieved source excerpts** to show page evidence.
- Ask an unsupported question; the assistant should say it cannot find it in the uploaded document.
- Replace the paper and show that only the new paper is used.
- Try an empty, password-protected, scanned, or unsupported file to demonstrate clear error handling.

## Notes

- Scanned/image-only PDFs need OCR before they can be indexed by the local extractor.
- Google free-tier availability and request limits can change. If a quota error occurs during the demo, wait briefly before retrying.
- The default model is `gemini-3.5-flash`. If Google AI Studio reports that it is unavailable for your key, set `GEMINI_MODEL` in `.env` to a model shown in your AI Studio account.
