# RAGGuard

Experimental open-source RAG evaluation toolkit with transparent lexical baselines and an **optional** external LLM judge.

**Important:** Token overlap is not factual verification. The LLM judge is probabilistic and may be wrong. RAGGuard is not a security or safety guarantee.

## Install

Python 3.10+:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Python usage

```python
from ragguard import evaluate, precision_at_k
print(evaluate('Capital of France?', 'Paris', ['Paris is the capital of France.']))
print(precision_at_k(['doc1', 'doc2'], ['doc1'], 2))
```

## CLI

```bash
ragguard examples/sample.json
ragguard examples/sample.jsonl --batch --output results.jsonl
```

## FastAPI

```bash
python -m pip install -e ".[api]"
uvicorn ragguard.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.
`POST /evaluate` accepts one record; `POST /evaluate/batch` accepts up to 100 records.

## Optional external LLM faithfulness judge

Requires a paid or separately provisioned API key and an OpenAI-compatible HTTPS chat completions endpoint. This feature sends your answer and retrieved context to that provider; don't send confidential data.

PowerShell:

```powershell
$env:RAGGUARD_API_KEY = 'YOUR_API_KEY'
ragguard examples/sample.json --judge-model YOUR_MODEL_NAME
Remove-Item Env:RAGGUARD_API_KEY
```

The returned `supported` boolean and unsupported claims are **model judgments, not verified facts**. Costs and availability depend on your provider. The optional judge is disabled unless `--judge-model` is supplied.

## Limitations

- `groundedness_proxy` measures overlap of unique answer tokens with context tokens, not truth.
- `question_overlap_f1` measures lexical overlap with the question, not semantic answer relevance.
- Precision@K uses `k` as denominator even when fewer than `k` items are returned.
- Batch JSONL validation fails on malformed records rather than silently skipping them.
- This is a pre-release; evaluate results manually before relying on them.

## Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [CHANGELOG.md](CHANGELOG.md).

MIT licensed. Project authorship and maintainer claims should reflect actual contributions.
