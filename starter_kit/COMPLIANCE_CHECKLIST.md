# Participant Guide Compliance Checklist

Source of truth:

- `https://platform.agentic-challenge.ai/files/participant_guide.html`
- local companion docs: `starter_kit/API.md`, `starter_kit/EVALUATION.md`

Current benchmark runner:

- [hackaton_difc_runner.py](/home/mkgs/hackaton/starter_kit/examples/hackaton_difc_runner.py)
- default answer model: `gpt-4.1-mini`
- retrieval stack: Docling parsing + local `multilingual-e5` + local Jina reranker + `dense_doc_diverse`

## Mandatory requirements

- `Only public APIs and public models may be used`
  - Status: satisfied
  - OpenAI `gpt-5-mini` is a public API model.
  - Local retrieval models are public Hugging Face models.

- `Telemetry required for every answer`
  - Status: satisfied
  - Every `SubmissionAnswer` includes `timing`, `usage`, `retrieval`, `model_name`.

- `No manual answering or partial automation`
  - Status: satisfied
  - Questions are fetched through the platform API and answered end-to-end by the runner.

- `No leaking or sharing private questions`
  - Status: satisfied
  - The runner only consumes the phase corpus locally and builds `submission.json`.
  - Code archive excludes live secrets and downloaded corpora.

## Submission-format requirements

- `architecture_summary` present
  - Status: satisfied

- Every answer contains `question_id`, `answer`, `telemetry`
  - Status: satisfied

- Deterministic absent answer -> JSON `null`
  - Status: satisfied

- Free-text answers are concise and capped
  - Status: satisfied
  - Runner truncates free-text answers to `280` characters.

- Unanswerable questions must use empty `retrieved_chunk_pages`
  - Status: satisfied
  - Runner now returns `[]` for:
    - deterministic `null`
    - free-text absence statements

## Retrieval / grounding requirements

- `retrieved_chunk_pages` included for answerable questions
  - Status: satisfied
  - Pages are normalized through `normalize_retrieved_pages(...)`.

- `doc_id` must match platform PDF IDs
  - Status: satisfied
  - Retrieval refs use the PDF filename stem hash from the downloaded corpus.

- `page_numbers` must be physical 1-based PDF pages
  - Status: satisfied
  - Retrieval refs are built from stored page metadata and normalized.

## Operational requirements

- Pull-model workflow (`GET /questions`, `GET /documents`, local run, `POST /submissions`)
  - Status: satisfied

- Code archive provided and secrets not shipped
  - Status: satisfied
  - Archive contains `.env.example`, not live `.env` or root `env`.

## Non-blocking gaps

- TTFT optimization
  - Status: not optimal, but compliant
  - Current runner uses non-streaming fallback:
    - `ttft_ms = total_time_ms`
    - `tpot_ms = 0`
  - This matches the starter-kit guidance for non-streaming APIs, but it is not leaderboard-optimal.

- Retrieval quality
  - Status: compliant, still being improved
  - This affects score quality, not submission validity.
