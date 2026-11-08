import argparse
import json
from pathlib import Path
from .metrics import evaluate
from .batch import evaluate_jsonl
from .judge import judge_faithfulness

def main():
    parser = argparse.ArgumentParser(description='RAGGuard RAG evaluation')
    parser.add_argument('input', type=Path, help='JSON file (or JSONL with --batch)')
    parser.add_argument('--batch', action='store_true', help='Evaluate JSONL records')
    parser.add_argument('--output', type=Path, help='Required output path for batch mode')
    parser.add_argument('--judge-model', help='Opt in to paid external LLM judging (single JSON only)')
    parser.add_argument('--base-url', default='https://api.openai.com/v1')
    args = parser.parse_args()
    if args.batch:
        if args.judge_model:
            parser.error('--judge-model cannot be combined with --batch')
        if not args.output:
            parser.error('--output is required with --batch')
        print(json.dumps({'evaluated':evaluate_jsonl(args.input,args.output),'output':str(args.output)}))
        return
    if args.output:
        parser.error('--output is supported only with --batch')
    data = json.loads(args.input.read_text(encoding='utf-8'))
    result = evaluate(**data)
    if args.judge_model:
        result['llm_judge'] = judge_faithfulness(data['answer'],data['contexts'],
                                                  model=args.judge_model,base_url=args.base_url)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
