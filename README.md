# TreeIntel Species Identifier — Web App

A small Flask web application that wraps your trained Keras (`.h5`) image
classification model in a browser-based upload page. Users drop in a photo
of a specimen and get back the predicted species with a confidence score
and the top-5 alternative matches.

## Project structure

```
treeintel_web_app/
├── app.py                # Flask backend (loads model once, serves /predict)
├── requirements.txt
├── models/
│   └── my_image_classifier_model.h5   <- put your trained model here
├── templates/
│   └── index.html        # Upload page markup
├── static/
│   ├── style.css          # Specimen-card styling
│   └── script.js          # Upload/drag-drop + fetch() calls to /predict
└── uploads/                # (optional) scratch folder, not required at runtime
```

## Setup

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   # source venv/bin/activate  # macOS/Linux
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your model file.** Copy your trained model into the `models/`
   folder so the path matches:
   ```
   treeintel_web_app/models/my_image_classifier_model.h5
   ```
   (This replaces the old hardcoded `C:\Users\...` path — the app now
   looks for the model relative to the project folder, which works the
   same on any machine or server.)

4. **Run the app:**
   ```bash
   python app.py
   ```

5. Open **http://127.0.0.1:5000** in your browser, upload a specimen
   photo, and click **Identify Specimen**.

## AI Knowledge (RAG + LLM)

The CNN's predicted species now feeds a retrieval-augmented LLM layer that
generates a botanical brief and answers follow-up questions, scoped to that
one species' documentation only.

### One-time setup

1. **Install the new dependencies** (already in `requirements.txt`):
   ```bash
   pip install -r requirements.txt
   ```

2. **Get a free Groq API key** at https://console.groq.com, then:
   ```bash
   cp .env.example .env
   ```
   and put your real key in `.env`. **Never commit `.env` or hardcode a key
   in source** — add `.env` to `.gitignore`.

3. **Add your species documents.** Drop your `.docx` write-ups into
   `knowledge_docs/`, then open `species_map.json` and make sure each CNN
   class name points at the right filename. You mentioned having 49 files
   for 51 classes — set the two without real documentation to `null` (the
   engine skips those cleanly instead of guessing).

4. **Build the vector store:**
   ```bash
   python scripts/build_knowledge_base.py
   ```
   Re-run this any time you add or edit a document. It prints exactly which
   classes loaded and which were skipped, so mapping mistakes are easy to
   spot.

5. **Run the app as usual** (`python app.py`). After a prediction, the AI
   Knowledge panel automatically requests `/knowledge/overview` for the
   predicted species and fills in; a chat box underneath lets you ask
   follow-up questions via `/knowledge/ask`, scoped to that species only.

### How retrieval stays scoped to one species

Every chunk is tagged with `metadata["species"]` at ingestion time, and both
`/knowledge/overview` and `/knowledge/ask` filter retrieval on that field —
so a question about one species can never surface passages from another.

### Notes

- The structured overview (description / uses / ecology / conservation) is
  generated once per species and cached to `knowledge_base/overview_cache/`
  — it won't re-call the LLM on every visit.
- Chat answers are not cached, since they're arbitrary follow-up questions.
- If a species has no mapped document, or the vector store hasn't been
  built yet, the panel says so explicitly rather than fabricating content —
  same principle as the Carbon Intelligence panel.
- `knowledge_engine.py` intentionally avoids LangChain's
  `create_retrieval_chain` / `create_stuff_documents_chain` helpers (which
  live in a package that's moved around across recent LangChain versions)
  in favor of a manual retrieve → format → `llm.invoke()` call. Fewer
  moving parts, easier to debug.

## How it works

- `app.py` loads the model **once** at startup (not per-request, which
  would be very slow) and exposes:
  - `GET /` — serves the upload page
  - `POST /predict` — accepts a multipart image upload (`image` field),
    runs inference, and returns JSON:
    ```json
    {
      "species": "Carapa_procera",
      "confidence": 92.14,
      "top5": [
        { "species": "Carapa_procera", "confidence": 92.14 },
        { "species": "Sp_6 Irvigia", "confidence": 4.02 },
        ...
      ]
    }
    ```
  - `GET /health` — simple check that the model loaded successfully

- The frontend (`static/script.js`) handles drag-and-drop or click-to-browse
  uploads, shows an image preview, calls `/predict` with `fetch()`, and
  renders the result as a specimen label card.

## Notes on your `class_names` list

Your original list mixes camera-folder codes (`105CANON`, `106CANON`, ...)
with real species names. I kept it exactly as-is so predictions still map
correctly to your trained model's output indices — **this list's order
must match the order Keras used during training** (usually alphabetical
folder order). If you ever retrain the model or rename folders, regenerate
this list from `train_generator.class_indices` to avoid silently mislabeled
predictions.

## Deploying beyond localhost

For a production deployment (e.g., hosted on a server rather than your own
machine):

- Run behind a WSGI server such as **Gunicorn** instead of Flask's dev
  server: `gunicorn -w 2 -b 0.0.0.0:8000 app:app`
  (keep `-w` workers low/tested — each worker loads its own copy of the
  TensorFlow model into memory).
- Store the `.h5` model in a persistent location the server can read
  (bundled in the deployment, or pulled from cloud storage like S3/GCS
  at container startup).
- Set `debug=False` (already the default outside `__main__` dev runs).
- Consider adding basic rate limiting if this will be publicly accessible,
  since each prediction call costs CPU/GPU time.
