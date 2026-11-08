import pytest
from ragguard.metrics import token_f1, groundedness, precision_at_k, recall_at_k, evaluate

def test_f1():
    assert token_f1('red apple', 'red apple') == 1
    assert token_f1('', 'red') == 0

def test_groundedness():
    assert groundedness('Paris France', ['Paris is in France']) == 1
    assert groundedness('moon', ['Paris is in France']) == 0

def test_retrieval():
    assert precision_at_k(['a', 'b'], ['a'], 2) == .5
    assert recall_at_k(['a', 'b'], ['a', 'c'], 2) == .5
    with pytest.raises(ValueError):
        precision_at_k([], [], 0)

def test_evaluate():
    result = evaluate('Where?', 'Paris', ['Paris is in France'], 'Paris')
    assert result['reference_token_f1'] == 1
