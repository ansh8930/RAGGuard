"""Simple lexical baselines, not semantic or factual verification."""
import re
from collections import Counter

def tokens(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())

def token_f1(prediction: str, reference: str) -> float:
    a, b = Counter(tokens(prediction)), Counter(tokens(reference))
    common = sum((a & b).values())
    if not a or not b or not common:
        return 0.0
    p, r = common / sum(a.values()), common / sum(b.values())
    return 2 * p * r / (p + r)

def groundedness(answer: str, contexts: list[str]) -> float:
    """Fraction of unique answer tokens appearing in retrieved contexts.

    Lexical proxy only: high scores do NOT establish factual faithfulness.
    """
    answer_words = set(tokens(answer))
    if not answer_words:
        return 0.0
    context_words = set(tokens(' '.join(contexts)))
    return len(answer_words & context_words) / len(answer_words)

def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError('k must be positive')
    top = retrieved_ids[:k]
    return len(set(top) & set(relevant_ids)) / k

def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError('k must be positive')
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant)

def evaluate(question: str, answer: str, contexts: list[str], reference: str | None = None) -> dict:
    """Return transparent lexical baseline metrics for a single RAG answer."""
    result = {'groundedness_proxy': round(groundedness(answer, contexts), 4),
              'question_overlap_f1': round(token_f1(answer, question), 4)}
    if reference is not None:
        result['reference_token_f1'] = round(token_f1(answer, reference), 4)
    return result
