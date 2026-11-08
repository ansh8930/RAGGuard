# RAGGuard

A small, dependency-free Python library for **transparent lexical baseline metrics** in retrieval-augmented generation (RAG) pipelines. Includes an optional FastAPI endpoint, CLI, tests, and GitHub Actions.

> **Important:** The `groundedness_proxy` score measures token overlap with retrieved text. It does **not** detect hallucinations reliably or verify factual correctness. `question_overlap_f1` is not a semantic answer relevance metric. These baselines should be supplemented with human evaluation or stronger evaluators before making quality claims.

## Quick start

```bash
python -m pip install -e '.[api,dev]'
python -m pytest -q
ragguard examples/sample.json
uvicorn ragguard.api:app --reload
```

POST to `http://127.0.0.1:8000/evaluate`:

```json
{"question":"What is the capital of France?","answer":"Paris","contexts":["Paris is the capital of France."],"reference":"Paris"}
```

## Python usage

```python
from ragguard import evaluate, precision_at_k, recall_at_k
print(evaluate('Capital of France?', 'Paris', ['Paris is in France'], 'Paris'))
print(precision_at_k(['doc1','doc2'], ['doc1'], 2))
print(recall_at_k(['doc1','doc2'], ['doc1','doc3'], 2))
```

## Metrics

- `groundedness_proxy`: fraction of distinct answer tokens also appearing in contexts; lexical overlap only.
- `question_overlap_f1`: bag-of-words token F1 against the question; not semantic relevance.
- `reference_token_f1`: token F1 against a provided reference answer.
- `precision_at_k`: number of unique relevant IDs among first k results divided by k.
- `recall_at_k`: number of unique relevant IDs among first k results divided by number of unique relevant IDs.

## Roadmap

- Dataset-level aggregate evaluation and JSONL input
- More robust evidence attribution
- Optional semantic scoring with documented model limitations
- CI quality threshold checks

## Contributing

Open an issue describing a bug or feature. Submit a focused PR with tests. Please don't claim this repository has community adoption until it does.

## License

MIT. See LICENSE.
