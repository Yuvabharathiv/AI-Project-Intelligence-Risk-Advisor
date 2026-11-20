# Milestone 4 — Project Insights and Risk Advisor

## Delivered

- **Insights dashboard:** Presents a rule-based health score and its component
  rationales, scope and delivery commitments, evidence-linked risks, blockers,
  actions, recommendations, and source-level milestone counts.
- **Incremental uploads:** New and changed PDF, DOCX, CSV, and TXT files are
  indexed and analyzed automatically. SHA-256 content hashes prevent
  reprocessing unchanged files. Updating an existing filename atomically
  replaces its chunks and analysis without deleting other project sources.
- **Persistent findings:** The local SQLite knowledge base stores analyses and
  content hashes alongside its vector index. Findings are restored on later
  app sessions and report outputs are invalidated when an uploaded source
  changes.
- **Grounded reports and chat:** The report includes scope, user-story
  templates, risks, blockers, actions, recommendations, and health rationale.
  Conversational responses continue to quote retrieved evidence and cite
  source filenames.
- **Operational guidance:** See [ARCHITECTURE.md](./ARCHITECTURE.md) for
  system details, scoring rules, setup, and known limitations.

## Verification

Run the full automated suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests exercise PDF, DOCX, CSV, and TXT ingestion through analysis, documentation
generation, vector retrieval, and cited Q&A. They also verify that identical
uploads are skipped, changed files replace only their own indexed source,
updated analyses persist across store instances, and unsupported files return
visible errors without altering the knowledge base. The automated workflow
uses fixed embeddings; it does not download or validate the real
sentence-transformer model.

## Final demonstration walkthrough

1. Start the app with `streamlit run app.py` and open **Documents & knowledge
   base**.
2. Upload a project plan or meeting note in PDF, DOCX, CSV, or TXT format.
   Show that the knowledge base indexes and analyzes it automatically.
3. Open **Dashboard** and review the scope summary, health dimensions,
   evidence-linked risk signals, recommended follow-ups, and action items.
4. Open **AI project assistant** and ask, “What are the biggest project risks?”
   Point out the cited source and quoted passage; ask an unsupported question
   to show that the assistant declines to invent an answer.
5. Upload an updated progress note. Show that only changed content is
   reprocessed and the combined dashboard reflects the latest analysis while
   the original sources remain indexed.
6. Open **Documentation generator**, generate the report, and download the
   Markdown output with source-linked scope, stories, risks, actions, blockers,
   recommendations, and scoring rationale.

## Known limitations

Analysis and health remain transparent keyword/rule-based signals, not
probability estimates or predictions. The agents can miss differently phrased
facts. Task completion, risk probability/impact, and historical health trends
are not inferred. Files are identified in the index by filename, so uploading
different files with the same filename intentionally replaces that source.
