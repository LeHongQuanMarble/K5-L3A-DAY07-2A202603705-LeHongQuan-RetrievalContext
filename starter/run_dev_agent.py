"""Chạy agent Claude trên một split với retriever và prompt đã chọn.

Mặc định (dense B, prompt v1, budget 1200) là cấu hình đã khóa trên test; kết quả ghi vào
starter/results/agent-<split>/. Các biến thể khác dùng --tag để không ghi đè kết quả cũ.

Dense cần biến môi trường DAY6_STARTER trỏ tới thư mục starter/ của repo Day 6 (venv của Day 6 có đủ thư viện):

  DAY6_STARTER=<day6>/starter <day6>/starter/.venv/bin/python starter/run_dev_agent.py --split dev

TF-IDF không cần Day 6 (vẫn cần anthropic và python-dotenv trong môi trường đang chạy).
Key và model lấy từ starter/.env.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent  # starter/ của Day 7
TOP_K = 3
STRATEGY = 'B'


def main():
    parser = argparse.ArgumentParser(description='Agent Claude trên một split')
    parser.add_argument('--split', required=True, choices=('dev', 'test'))
    parser.add_argument('--retriever', choices=('dense', 'tfidf'), default='dense')
    parser.add_argument('--prompt', choices=('v1', 'v2'), default='v1')
    parser.add_argument('--budget', type=int, default=1200, help='Ký tự context; cấu hình khóa: 1200 (dense), 800 (tfidf)')
    parser.add_argument('--tag', default='', help='Hậu tố thư mục kết quả, ví dụ "v2" -> agent-dev-v2')
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    from anthropic import Anthropic
    from agentic import ClaudeController, DenseRetriever, TfidfRetriever, run_agent

    env = dotenv_values(ROOT / '.env')
    data = ROOT / 'data' / 'converted'
    chunks = json.loads((data / 'chunks.json').read_text(encoding='utf-8'))
    questions = {q['id']: q for q in json.loads((data / 'questions.json').read_text(encoding='utf-8'))}
    # Câu thiếu evidence nằm ở data/pride_unanswerable.json, không nằm trong questions.json.
    unanswerable = {u['id']: u for u in json.loads((ROOT.parent / 'data' / 'pride_unanswerable.json').read_text(encoding='utf-8'))}
    cases = [c for c in json.loads((data / 'cases.json').read_text(encoding='utf-8')) if c['split'] == args.split]

    if args.retriever == 'dense':
        if 'DAY6_STARTER' not in os.environ:
            raise SystemExit('Đặt biến môi trường DAY6_STARTER trỏ tới thư mục starter/ của repo Day 6')
        day6 = Path(os.environ['DAY6_STARTER']).resolve()
        sys.path.insert(0, str(day6))
        from src.embeddings import SentenceTransformerEmbedder
        from src.store import PgVectorStore
        embedder = SentenceTransformerEmbedder()
        store = PgVectorStore(dotenv_values(day6 / '.env')['DATABASE_URL'], dimensions=embedder.dimensions)
        retriever = DenseRetriever(store, embedder, chunks, strategy=STRATEGY)
    else:
        retriever = TfidfRetriever(chunks)

    controller = ClaudeController(env['MODEL'], client=Anthropic(api_key=env['ANTHROPIC_API_KEY']), prompt=args.prompt)

    traces = []
    wall_start = time.perf_counter()
    for case in cases:
        filters = questions.get(case['id'], {}).get('filter')
        if case['id'] in unanswerable:
            filters = unanswerable[case['id']].get('filter')
        started = time.perf_counter()
        try:
            result = run_agent(case['question'], controller, retriever, filters=filters, top_k=TOP_K, budget=args.budget)
            traces.append({'query_id': case['id'], **result})
        except Exception as exc:  # lỗi được ghi riêng, không tính vào chất lượng
            traces.append({'query_id': case['id'], 'status': 'error', 'error': repr(exc), 'events': [], 'citations': []})
        traces[-1]['elapsed_ms'] = (time.perf_counter() - started) * 1000

    name = f'agent-{args.split}' + (f'-{args.tag}' if args.tag else '')
    out = ROOT / 'results' / name
    out.mkdir(parents=True, exist_ok=True)
    config = {'split': args.split, 'retriever': args.retriever, 'model': env['MODEL'],
              'strategy': STRATEGY if args.retriever == 'dense' else 'tfidf', 'prompt': args.prompt,
              'top_k': TOP_K, 'budget_chars': args.budget, 'max_searches': 4, 'n': len(traces),
              'wall_seconds': round(time.perf_counter() - wall_start, 1), 'usage': controller.usage}
    (out / 'traces.json').write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
