"""Thử nghiệm: truy xuất trang (hybrid, cùng index), rồi gửi ảnh các trang đó cho Claude (vision) để trả lời.

Mục đích: kiểm tra xem nút thắt là truy xuất text hay khả năng đọc bảng/biểu đồ. Không dùng gold trong query hay trong prompt.
Model chỉ được trả lời từ ảnh trang; citation là số trang, phải thuộc các trang đã truy xuất.

  <day6>/starter/.venv/bin/python starter/advanced/vision_eval.py --split dev --pages 3 --tag vision-h3
"""
import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path

from dotenv import dotenv_values

HERE = Path(__file__).resolve().parent
STARTER = HERE.parent
sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(HERE))

from build_index import render_page  # noqa: E402  (cùng cách render với index)
from run_advanced import AdvRetriever, INDEX, load_benchmark  # noqa: E402

PROMPT = (
    'Bạn trả lời câu hỏi về Annual Energy Review 2011 chỉ dựa trên các trang ảnh đính kèm (đánh số theo trang PDF). '
    'Đọc kỹ bảng, hình và chú thích. Nếu trả lời được, nêu đúng năm, hàng/cột, đơn vị hoặc tên hình. '
    'Nếu ảnh không có thông tin, trả lời đúng chuỗi "KHONG_DU_BANG_CHUNG". '
    'Trả về duy nhất một JSON: {"answer": "...", "pages": [số trang dùng để trả lời]}.\n\nCâu hỏi: '
)


def parse(text):
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=('dev', 'test', 'test2'))
    ap.add_argument('--pages', type=int, default=3)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--only', default='')
    ap.add_argument('--pager', default='hybrid', choices=('hybrid', 'page_rrf'))
    args = ap.parse_args()

    import numpy as np
    from anthropic import Anthropic
    from sentence_transformers import SentenceTransformer

    chunks = json.loads((INDEX / 'chunks.json').read_text(encoding='utf-8'))
    vectors = np.load(INDEX / 'vectors.npy')

    class Emb:
        def __init__(self):
            self.m = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            self.m.max_seq_length = 256

        def embed_query(self, q):
            return self.m.encode([q], normalize_embeddings=True, show_progress_bar=False)[0].tolist()

    retriever = AdvRetriever(chunks, 'hybrid', Emb(), vectors)
    if args.pager == 'page_rrf':
        from page_rank import PageRanker
        ranker = PageRanker(chunks, vectors, retriever.embedder)
    env = dotenv_values(STARTER / '.env')
    client = Anthropic(api_key=env['ANTHROPIC_API_KEY'])

    items = load_benchmark(args.split)
    if args.only:
        keep = set(args.only.split(','))
        items = [x for x in items if x['id'] in keep]
    traces, usage = [], {'calls': 0, 'input_tokens': 0, 'output_tokens': 0}
    wall = time.perf_counter()
    for item in items:
        started = time.perf_counter()
        if args.pager == 'page_rrf':
            results = [{'id': None, 'page': p} for p in ranker.rank(item['question'], 'page_rrf')]
            pages = [r['page'] for r in results[:args.pages]]
        else:
            results = retriever.search(item['question'], top_k=12)
            pages = []
            for r in results:  # trang duy nhất theo thứ tự xếp hạng
                if r['page'] not in pages:
                    pages.append(r['page'])
                if len(pages) >= args.pages:
                    break
        content = []
        for p in pages:
            png = render_page(p)
            content.append({'type': 'text', 'text': f'Trang {p}:'})
            content.append({'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png',
                                                       'data': base64.b64encode(png.read_bytes()).decode()}})
        content.append({'type': 'text', 'text': PROMPT + item['question']})
        resp = client.messages.create(model=env['MODEL'], max_tokens=1500,
                                      messages=[{'role': 'user', 'content': content}])
        usage['calls'] += 1
        usage['input_tokens'] += resp.usage.input_tokens
        usage['output_tokens'] += resp.usage.output_tokens
        text = ''.join(b.text for b in resp.content if b.type == 'text')
        parsed = parse(text) or {}
        answer = str(parsed.get('answer', text)).strip()
        cited = [p for p in parsed.get('pages', []) if isinstance(p, int) and p in pages]
        abstain = (not answer) or ('KHONG_DU_BANG_CHUNG' in answer) or not cited
        status = 'abstain' if abstain else 'answered'
        traces.append({
            'query_id': item['id'], 'group': item['group'], 'status': status,
            'answer': 'Không đủ bằng chứng để trả lời.' if abstain else answer,
            'raw_answer': text, 'citations': [f'page:{p}' for p in cited],
            'events': [{'event': 'search', 'query': item['question'],
                        'results': [{'id': r['id'], 'page': r['page']} for r in results[:args.pages * 4]]},
                       {'event': 'answer', 'pages_shown': pages}],
            'elapsed_ms': (time.perf_counter() - started) * 1000,
        })
        print(item['id'], status, pages, flush=True)

    out = HERE / 'results' / f'{args.split}-{args.tag}'
    out.mkdir(parents=True, exist_ok=True)
    config = {'split': args.split, 'mode': 'vision_pages', 'page_selection': args.pager, 'pages_shown': args.pages,
              'model': env['MODEL'], 'n': len(traces), 'wall_seconds': round(time.perf_counter() - wall, 1),
              'usage': usage}
    (out / 'traces.json').write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(config, ensure_ascii=False))


if __name__ == '__main__':
    main()
