Public eval files in this folder should be sanitized and synthetic.

Guidelines:
- Do not commit internal reference answers copied from proprietary documents.
- Do not commit source file names/chunk ids that reveal private corpus structure.
- Keep private golden sets under `eval/private/` (gitignored).

Recommended split:
- `eval/golden_qna.jsonl`: public keyword-only checks.
- `eval/splits/*.jsonl`: public lightweight checks without private citation targets.
- `eval/private/*.jsonl`: internal full-fidelity golden sets with reference answers/citations.
