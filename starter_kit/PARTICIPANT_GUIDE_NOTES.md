# Agentic Challenge Notes

Official references:

- Platform: `https://platform.agentic-challenge.ai/`
- Participant guide: `https://platform.agentic-challenge.ai/files/participant_guide.html`
- Starter kit docs: `starter_kit/README.md`, `starter_kit/API.md`, `starter_kit/EVALUATION.md`

## What the platform expects

- This is an API-only competition.
- Participants do not host a service for the organizers.
- The required flow is:
  1. download questions via `GET /questions`
  2. download the current phase corpus via `GET /documents`
  3. run the pipeline locally or on participant infrastructure
  4. upload `submission.json` and a code ZIP via `POST /submissions`

## Phase rules

- Each phase has its own corpus and it must be indexed independently.
- Warm-up scale is about `100` questions on about `30` documents.
- Final scale is about `900` questions on about `300` documents.
- Documents are heterogeneous PDFs, including scanned pages, so OCR-capable ingestion is required.

## Submission format

- File-level fields:
  - `architecture_summary`
  - `answers`
- Each answer must contain:
  - `question_id`
  - `answer`
  - `telemetry`

## Answer-type rules

- `number`: JSON number, scored with about `±1%` tolerance
- `boolean`: JSON `true` / `false`
- `name`: single exact normalized string
- `names`: JSON array of strings
- `date`: ISO `YYYY-MM-DD`
- `free_text`: concise grounded answer, expected max about `280` characters

For deterministic types, `null` is a valid answer when the information is absent from the corpus.

For unanswerable `free_text`, return a natural-language absence statement and set retrieved pages to `[]`.

## Telemetry rules

- `telemetry` is required for every answer.
- `doc_id` must be the PDF filename stem used by the platform corpus.
- `page_numbers` must be physical PDF pages, 1-based.
- Include only pages actually used for the final answer, not every retrieved page.
- If the answer is truly unanswerable, `retrieved_chunk_pages` should be `[]`.
- Missing or malformed telemetry reduces the telemetry factor.

## Scoring summary

The platform combines:

- deterministic score
- assistant score
- grounding score
- telemetry factor
- TTFT factor

The documented formula is:

`Total = (0.7 * deterministic + 0.3 * assistant) * grounding * telemetry * ttft_factor`

Grounding is the most important multiplier, so over-reporting pages is harmful.

## Reproducibility rules

- If external APIs are used, they must be public and reproducible.
- Local indexing/storage is allowed.
- Real secrets must not be shipped in the code archive.
- The archive should include `.env.example`, not live keys.

## Current local readiness

- `EVAL_API_KEY` is stored in root `env` and `starter_kit/.env`.
- A challenge runner was prepared in `starter_kit/examples/hackaton_difc_runner.py`.
- The runner is designed to:
  - download phase resources via `arlc.EvaluationClient`
  - parse/index the corpus with the current DIFC RAG pipeline
  - build competition-format telemetry
  - save `submission.json`
  - optionally submit it with a code archive

Warm-up API verification already completed:

- `GET /questions` returned `100` warm-up questions
- `GET /documents` returned `30` PDFs
- the warm-up corpus was ingested locally into `starter_kit/challenge_workdir`
- current parsed corpus size: `30` documents, `1996` chunks

Important dataset difference:

- the platform warm-up questions are **not** identical to the local `public_dataset.json`
- the warm-up corpus overlaps heavily with the local `pdfs/` corpus, but it is not identical either
- therefore local public-dataset metrics are useful for development, but they are **not** a direct benchmark for the platform warm-up set

The only missing ingredient for a full benchmark run is a working answer-model key, which can be supplied later.
