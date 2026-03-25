# Benchmark Run Notes

Date: `2026-03-11`

## Final choice

- answer model: `gpt-4.1-mini`
- retrieval stack: `Docling + multilingual-e5 + local Jina reranker + dense_doc_diverse`
- platform flow: `GET /questions` + `GET /documents` + local run + local `submission.json`

## Why `gpt-5-mini` was dropped

- `gpt-5-mini` through the current `chat.completions` path spent completion budget on reasoning tokens.
- Result: `message.content` frequently came back empty.
- This produced schema-valid but low-quality outputs:
  - deterministic answers collapsed to `null`
  - free-text answers collapsed to absence strings
- It was also much slower, with per-question latency around `10s` to `15s+`.

## `gpt-4.1-mini` benchmark result

- `100 / 100` answers generated
- `0` runtime errors in `submission_debug.json`
- local submission validator: `ok = true`
- code archive size: `355,088` bytes

Timing summary from submission telemetry:

- mean `total_time_ms`: `1744.2`
- median `total_time_ms`: `1640`
- max `total_time_ms`: `3972`
- min `total_time_ms`: `830`

Answer distribution:

- `1` null answer total
- `2` answers with empty retrieval refs

## Compliance status

- checked against participant guide and starter-kit API/evaluation docs
- local checklist: [COMPLIANCE_CHECKLIST.md](/home/mkgs/hackaton/starter_kit/COMPLIANCE_CHECKLIST.md)
- local validator: [validate_submission_local.py](/home/mkgs/hackaton/starter_kit/validate_submission_local.py)

## Output artifacts

- submission: [submission.json](/home/mkgs/hackaton/starter_kit/submission.json)
- debug rows: [submission_debug.json](/home/mkgs/hackaton/starter_kit/challenge_workdir/submission_debug.json)
- code archive: [code_archive.zip](/home/mkgs/hackaton/starter_kit/code_archive.zip)
