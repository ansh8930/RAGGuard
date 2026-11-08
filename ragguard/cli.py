import argparse
import json
from pathlib import Path
from .metrics import evaluate

def main():
    parser = argparse.ArgumentParser(description='Evaluate a RAG response from JSON')
    parser.add_argument('input', type=Path, help='JSON containing question, answer, contexts, optional reference')
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding='utf-8'))
    print(json.dumps(evaluate(**data), indent=2))

if __name__ == '__main__':
    main()
