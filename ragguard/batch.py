"""Batch processing of JSONL datasets; input is one evaluation object per line."""
import json
from pathlib import Path
from .metrics import evaluate

def evaluate_jsonl(input_path: str | Path, output_path: str | Path) -> int:
    source, target = Path(input_path), Path(output_path)
    if source.resolve() == target.resolve():
        raise ValueError('input and output must be different paths')
    rows = []
    for line_no, line in enumerate(source.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if not isinstance(item,dict):
                raise ValueError('expected JSON object')
            rows.append({'line':line_no, 'metrics':evaluate(**item)})
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f'Invalid JSONL record at line {line_no}: {exc}') from exc
    target.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows), encoding='utf-8')
    return len(rows)
