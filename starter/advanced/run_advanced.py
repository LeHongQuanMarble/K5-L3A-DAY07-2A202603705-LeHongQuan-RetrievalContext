"""Chạy agent trên benchmark nâng cao (Annual Energy Review 2011) với retriever đã chọn.

Retriever (cùng index, cùng corpus, cùng chunk):
  dense   — MiniLM đa ngôn ngữ (cùng model Day 6), cosine trên vectors.npy
  tfidf   — lexical TF-IDF (baseline.TfidfIndex)
  hybrid  — Reciprocal Rank Fusion của dense và tfidf (k=60, 20 ứng viên mỗi nguồn)

Agent dùng agentic.run_agent và ClaudeController (prompt v1 hoặc v2). Không đưa gold vào query hay context.

  <day6>/starter/.venv/bin/python starter/advanced/run_advanced.py --split dev --method dense --top-k 3 --budget 2500 --prompt v2 --tag d1

Kết quả ghi vào starter/advanced/results/<split>-<tag>/ (traces.json, config.json).
"""
import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import dotenv_values

HERE = Path(__file__).resolve().parent  # starter/advanced
STARTER = HERE.parent  # starter/
sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(HERE))

INDEX = STARTER / 'data' / 'advanced' / 'index'
BENCH = HERE / 'benchmark'
RRF_K = 60
CANDIDATES = 20


class AdvRetriever:
    def __init__(self, chunks, method, embedder=None, vectors=None):
        from baseline import TfidfIndex
        self.chunks = chunks
        self.by_id = {c['id']: c for c in chunks}
        self.method = method
        self.embedder = embedder
        self.vectors = vectors
        self.tfidf = TfidfIndex(chunks) if method in ('tfidf', 'hybrid') else None
        self.seen = {}

    def _dense(self, query, n):
        import numpy as np
        q = np.asarray(self.embedder.embed_query(query), dtype='float32')
        scores = self.vectors @ q
        order = np.argsort(-scores)[:n]
        return [(self.chunks[i]['id'], float(scores[i])) for i in order]

    def _lexical(self, query, n):
        return [(c['id'], c['score']) for c in self.tfidf.search(query, top_k=n)]

    def _ranking(self, query, n):
        if self.method == 'dense':
            return self._dense(query, n)
        if self.method == 'tfidf':
            return self._lexical(query, n)
        dense = self._dense(query, CANDIDATES)
        lexical = self._lexical(query, CANDIDATES)
        fused = {}
        for ranking in (dense, lexical):
            for rank, (cid, _) in enumerate(ranking, start=1):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:n]

    def search(self, query, *, filters=None, top_k=3):
        # Query lặp lại: trả trang kết quả kế tiếp (hạng top_k+1 trở đi) thay vì cùng một kết quả.
        key = ' '.join(query.casefold().split())
        repeats = self.seen.get(key, 0)
        self.seen[key] = repeats + 1
        n = top_k * (repeats + 1)
        ranked = self._ranking(query, max(n, CANDIDATES))
        page = ranked[repeats * top_k:(repeats + 1) * top_k]
        return [{**self.by_id[i], 'score': s} for i, s in page]


def load_benchmark(split):
    items = json.loads((BENCH / f'{split}.json').read_text(encoding='utf-8'))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, choices=('dev', 'test'))
    parser.add_argument('--method', choices=('dense', 'tfidf', 'hybrid'), default='dense')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--budget', type=int, default=2500)
    parser.add_argument('--prompt', choices=('v1', 'v2', 'v3'), default='v2')
    parser.add_argument('--max-searches', type=int, default=4)
    parser.add_argument('--pack', choices=('insertion', 'score'), default='score')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--only', default='', help='Chỉ chạy các ID này (phẩy ngăn cách), để debug')
    args = parser.parse_args()

    from anthropic import Anthropic
    from agentic import ClaudeController, run_agent

    chunks = json.loads((INDEX / 'chunks.json').read_text(encoding='utf-8'))
    import numpy as np
    vectors = np.load(INDEX / 'vectors.npy')
    assert len(chunks) == vectors.shape[0]

    embedder = None
    if args.method in ('dense', 'hybrid'):
        from sentence_transformers import SentenceTransformer

        class Emb:
            def __init__(self):
                self.m = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
                self.m.max_seq_length = 256

            def embed_query(self, q):
                return self.m.encode([q], normalize_embeddings=True, show_progress_bar=False)[0].tolist()

        embedder = Emb()
    retriever = AdvRetriever(chunks, args.method, embedder, vectors)

    env = dotenv_values(STARTER / '.env')
    controller = ClaudeController(env['MODEL'], client=Anthropic(api_key=env['ANTHROPIC_API_KEY']), prompt=args.prompt)

    items = load_benchmark(args.split)
    if args.only:
        keep = set(args.only.split(','))
        items = [x for x in items if x['id'] in keep]
    traces = []
    wall = time.perf_counter()
    for item in items:
        started = time.perf_counter()
        try:
            result = run_agent(item['question'], controller, retriever, filters=None,
                               top_k=args.top_k, budget=args.budget, max_searches=args.max_searches,
                               pack_order=args.pack)
            trace = {'query_id': item['id'], 'group': item['group'], **result}
        except Exception as exc:  # lỗi được ghi riêng, không tính vào chất lượng
            trace = {'query_id': item['id'], 'group': item['group'], 'status': 'error', 'error': repr(exc),
                     'events': [], 'citations': [], 'answer': ''}
        trace['elapsed_ms'] = (time.perf_counter() - started) * 1000
        traces.append(trace)
        print(item['id'], trace['status'], round(trace['elapsed_ms']), flush=True)

    out = HERE / 'results' / f'{args.split}-{args.tag}'
    out.mkdir(parents=True, exist_ok=True)
    config = {'split': args.split, 'method': args.method, 'top_k': args.top_k, 'budget_chars': args.budget,
              'prompt': args.prompt, 'max_searches': args.max_searches, 'pack': args.pack, 'model': env['MODEL'],
              'index': 'data/advanced/index (aer2011, text + figure descriptions)', 'n': len(traces),
              'wall_seconds': round(time.perf_counter() - wall, 1), 'usage': controller.usage}
    (out / 'traces.json').write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(config, ensure_ascii=False))


if __name__ == '__main__':
    main()
