"""Dense retrieval baseline on the Day 6 pgvector store (strategy B), same protocol as baseline.py.

Reuses Day 6 code for the embedding model and the store, and Day 7's evaluate()/pack_context()
so metrics are directly comparable with the TF-IDF run. Needs Day 6's environment (psycopg,
sentence-transformers) and the Day 6 database running.

  <day6>/starter/.venv/bin/python starter/dense_dev.py --day6 <day6>/starter \
    --chunks starter/data/converted/chunks.json --questions starter/data/converted/questions.json \
    --split dev --output starter/results/dev-dense-B
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from baseline import TfidfIndex, evaluate, validate  # noqa: E402

STRATEGY = 'B'  # Day 6 table strategy: paragraph groups, max 800 chars
TOP_K = 3
RRF_K = 60  # reciprocal rank fusion constant
CANDIDATES = 20  # candidates taken from each retriever before fusion


class DenseIndex:
    """Adapter with the same .search(question, top_k) interface as baseline.TfidfIndex.

    method='dense': pgvector cosine top-k.
    method='hybrid': reciprocal rank fusion of dense and TF-IDF candidate lists.
    """

    def __init__(self, store, embedder, chunks, method):
        self.store, self.embedder, self.method = store, embedder, method
        self.chunks_by_id = {c['id']: c for c in chunks}
        self.tfidf = TfidfIndex(chunks) if method == 'hybrid' else None

    def _dense(self, question, n):
        vector = self.embedder.embed_query(question)
        rows = self.store.search(STRATEGY, vector, top_k=n)
        return [(row['id'], float(row['score'])) for row in rows]

    def search(self, question, top_k=3):
        if self.method == 'dense':
            ranked = self._dense(question, top_k)
            return [{**self.chunks_by_id[i], 'score': s, 'strategy': 'dense-B'} for i, s in ranked]
        dense = self._dense(question, CANDIDATES)
        lexical = [(c['id'], c['score']) for c in self.tfidf.search(question, top_k=CANDIDATES)]
        fused = {}
        for ranking in (dense, lexical):
            for rank, (cid, _) in enumerate(ranking, start=1):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        best = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [{**self.chunks_by_id[i], 'score': s, 'strategy': 'hybrid-B-rrf'} for i, s in best]


def main():
    parser = argparse.ArgumentParser(description='Dense retrieval (Day 6 pgvector, strategy B) trên một split')
    parser.add_argument('--day6', required=True, type=Path, help='Thư mục starter/ của repo Day 6')
    parser.add_argument('--chunks', required=True, type=Path)
    parser.add_argument('--questions', required=True, type=Path)
    parser.add_argument('--split', required=True, choices=('dev', 'test'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--method', choices=('dense', 'hybrid'), default='dense')
    parser.add_argument('--budget', type=int, default=800, help='Ký tự context; 800 = mặc định baseline')
    args = parser.parse_args()

    day6 = args.day6.resolve()
    sys.path.insert(0, str(day6))
    from src.embeddings import SentenceTransformerEmbedder
    from src.store import PgVectorStore
    from dotenv import dotenv_values

    chunks = json.loads(args.chunks.read_text(encoding='utf-8'))
    questions = json.loads(args.questions.read_text(encoding='utf-8'))
    selected = [q for q in questions if q.get('split') == args.split]  # never touch the other split
    validate(chunks, selected)

    database_url = dotenv_values(day6 / '.env').get('DATABASE_URL')
    embedder = SentenceTransformerEmbedder()
    store = PgVectorStore(database_url, dimensions=embedder.dimensions)
    store.create_schema()  # only verifies the table; does not write rows
    if store.count(STRATEGY) != len(chunks):
        raise SystemExit(f'store has {store.count(STRATEGY)} rows for strategy {STRATEGY}, expected {len(chunks)}')

    started = time.perf_counter()
    index = DenseIndex(store, embedder, chunks, args.method)
    traces = evaluate(index, selected, budget=args.budget)
    name = 'dense-B' if args.method == 'dense' else 'hybrid-B-rrf'
    for t in traces:
        t['strategy'] = name
        t['top_k'] = TOP_K
    summary = {'strategy': name, 'budget_chars': args.budget, 'model': embedder.model_name, 'split': args.split, 'n': len(traces),
               'hit1': sum(t['hit1'] for t in traces), 'hit3': sum(t['hit3'] for t in traces),
               'context_hit': sum(t['context_hit'] for t in traces),
               'avg_context_chars': statistics.mean(t['context_chars'] for t in traces),
               'median_ms': statistics.median(t['elapsed_ms'] for t in traces),
               'total_seconds': time.perf_counter() - started, 'service_cost_usd': 0}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, obj in (('traces.json', traces), ('summary.json', summary)):
        (args.output / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
