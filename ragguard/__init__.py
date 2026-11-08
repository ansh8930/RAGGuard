"""RAGGuard: transparent lexical baselines and optional LLM judge."""
from .metrics import evaluate, precision_at_k, recall_at_k, token_f1, groundedness
from .batch import evaluate_jsonl
from .judge import judge_faithfulness, JudgeError
__all__ = ['evaluate','precision_at_k','recall_at_k','token_f1','groundedness',
           'evaluate_jsonl','judge_faithfulness','JudgeError']
