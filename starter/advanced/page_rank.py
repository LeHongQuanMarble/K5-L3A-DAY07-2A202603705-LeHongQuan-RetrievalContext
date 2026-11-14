"""Xếp hạng trang cho thử nghiệm vision. Các phương pháp chỉ dùng câu hỏi, không dùng gold.

  dense_chunk   : cosine chunk, lấy điểm cao nhất của mỗi trang
  tfidf_chunk   : TF-IDF chunk, điểm cao nhất của mỗi trang
  hybrid_chunk  : RRF chunk (dense + tfidf), xếp trang theo chunk đầu tiên của trang
  page_rrf      : RRF giữa thứ hạng trang dense_chunk và tfidf_chunk
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STARTER = HERE.parent
sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(HERE))
from baseline import TfidfIndex  # noqa: E402

INDEX = STARTER / 'data' / 'advanced' / 'index'
RRF_K = 60


class PageRanker:
    def __init__(self, chunks, vectors, embedder):
        self.chunks, self.vectors, self.embedder = chunks, vectors, embedder
        self.tfidf = TfidfIndex(chunks)
        self.page_of = np.array([c['page'] for c in chunks])
        self.pages = sorted(set(self.page_of.tolist()))

    def _dense_scores(self, q):
        qv = np.asarray(self.embedder.embed_query(q), dtype='float32')
        return self.vectors @ qv

    def _tfidf_scores(self, q):
        s = np.zeros(len(self.chunks), dtype='float32')
        for c in self.tfidf.search(q, top_k=len(self.chunks)):
            s[self.chunks_idx[c['id']]] = c['score']
        return s

    def _page_max(self, chunk_scores):
        best = defaultdict(lambda: -1e9)
        for i, s in enumerate(chunk_scores):
            p = self.page_of[i]
            if s > best[p]:
                best[p] = s
        return [int(p) for p, _ in sorted(best.items(), key=lambda kv: -kv[1])]

    def rank(self, q, method):
        if not hasattr(self, 'chunks_idx'):
            self.chunks_idx = {c['id']: i for i, c in enumerate(self.chunks)}
        if method == 'dense_chunk':
            return self._page_max(self._dense_scores(q))
        if method == 'tfidf_chunk':
            return self._page_max(self._tfidf_scores(q))
        if method == 'hybrid_chunk':
            d = np.argsort(-self._dense_scores(q))[:60]
            t = np.argsort(-self._tfidf_scores(q))[:60]
            fused = defaultdict(float)
            for ranking in (d, t):
                for r, i in enumerate(ranking, start=1):
                    fused[i] += 1.0 / (RRF_K + r)
            order = sorted(fused.items(), key=lambda kv: -kv[1])
            pages = []
            for i, _ in order:
                if self.page_of[i] not in pages:
                    pages.append(int(self.page_of[i]))
            return pages
        if method == 'page_rrf':
            a = self.rank(q, 'dense_chunk')
            b = self.rank(q, 'tfidf_chunk')
            fused = defaultdict(float)
            for ranking in (a, b):
                for r, p in enumerate(ranking, start=1):
                    fused[p] += 1.0 / (RRF_K + r)
            return [int(p) for p, _ in sorted(fused.items(), key=lambda kv: -kv[1])]
        raise ValueError(method)


if __name__ == '__main__':
    from sentence_transformers import SentenceTransformer

    class Emb:
        def __init__(self):
            self.m = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            self.m.max_seq_length = 256

        def embed_query(self, q):
            return self.m.encode([q], normalize_embeddings=True, show_progress_bar=False)[0].tolist()

    chunks = json.loads((INDEX / 'chunks.json').read_text(encoding='utf-8'))
    vectors = np.load(INDEX / 'vectors.npy')
    ranker = PageRanker(chunks, vectors, Emb())
    bench = [x for x in json.loads((HERE / 'benchmark' / 'dev.json').read_text(encoding='utf-8')) if x['gold']]
    for method in ('dense_chunk', 'tfidf_chunk', 'hybrid_chunk', 'page_rrf'):
        line = [method]
        for k in (6, 8, 10):
            hit = 0
            for x in bench:
                gold = {g['page'] for g in x['gold']}
                if gold & set(ranker.rank(x['question'], method)[:k]):
                    hit += 1
            line.append(f'R@{k}={hit}/{len(bench)}')
        print(*line, flush=True)
