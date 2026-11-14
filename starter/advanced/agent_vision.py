"""Agent vision nhiều vòng cho benchmark nâng cao (chỉ dùng để phát triển trên dev).

Mỗi vòng: model xem ảnh các trang đã có. Nó hoặc
  - trả lời: {"action": "answer", "answer": "...", "pages": [...]}, hoặc
  - yêu cầu thêm trang: {"action": "search", "queries": ["...", "..."]}.
Truy vấn mới dùng page_rrf (cùng index, không dùng gold). Tối đa MAX_ROUNDS vòng; hết vòng thì từ chối.
Tổng số trang gửi cho model mỗi vòng tăng dần, giới hạn MAX_PAGES.

  <day6>/starter/.venv/bin/python starter/advanced/agent_vision.py --split dev --tag agentv1
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

from build_index import render_page  # noqa: E402
from page_rank import PageRanker  # noqa: E402
from run_advanced import INDEX, load_benchmark  # noqa: E402

MAX_ROUNDS = 3
MAX_PAGES = 12
PER_QUERY = 3

SYSTEM = (
    'Bạn trả lời câu hỏi về Annual Energy Review 2011 chỉ dựa trên các trang ảnh đã cho (số là số trang PDF). '
    'Đọc kỹ bảng, hình và chú thích. Nếu đã có đủ bằng chứng, trả về JSON: '
    '{"action": "answer", "answer": "...", "pages": [số trang dùng]}. '
    'Nếu chưa đủ nhưng có thể tìm thêm, trả về JSON: {"action": "search", "queries": ["truy vấn tiếng Anh hoặc Việt, '
    'nêu năm, tên bảng/hình, hàng hoặc cột cần tìm", "..."]} (tối đa 2 truy vấn). '
    'Nếu không có thông tin và không thể tìm thêm, trả về {"action": "answer", "answer": "KHONG_DU_BANG_CHUNG", "pages": []}. '
    'Chỉ trả về JSON.\n\nCâu hỏi: '
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
    ap.add_argument('--split', required=True, choices=('dev', 'test', 'test2', 'test3'))
    ap.add_argument('--tag', required=True)
    ap.add_argument('--only', default='')
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

    ranker = PageRanker(chunks, vectors, Emb())
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
        pages = [int(p) for p in ranker.rank(item['question'], 'page_rrf')[:6]]
        events, rounds = [], 0
        final = None
        while True:
            rounds += 1
            content = []
            for p in pages:
                content.append({'type': 'text', 'text': f'Trang {p}:'})
                content.append({'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png',
                                                           'data': base64.b64encode(render_page(p).read_bytes()).decode()}})
            content.append({'type': 'text', 'text': SYSTEM + item['question']})
            resp = client.messages.create(model=env['MODEL'], max_tokens=1200,
                                          messages=[{'role': 'user', 'content': content}])
            usage['calls'] += 1
            usage['input_tokens'] += resp.usage.input_tokens
            usage['output_tokens'] += resp.usage.output_tokens
            text = ''.join(b.text for b in resp.content if b.type == 'text')
            parsed = parse(text) or {}
            events.append({'event': 'round', 'round': rounds, 'pages_shown': list(pages), 'raw': text[:400]})
            if parsed.get('action') == 'search' and rounds < MAX_ROUNDS and len(pages) < MAX_PAGES:
                new = []
                for q in (parsed.get('queries') or [])[:2]:
                    if not isinstance(q, str) or not q.strip():
                        continue
                    for p in ranker.rank(q, 'page_rrf')[:PER_QUERY]:
                        p = int(p)
                        if p not in pages and p not in new:
                            new.append(p)
                pages = pages + new[:MAX_PAGES - len(pages)]
                events[-1]['new_pages'] = new[:MAX_PAGES]
                continue
            final = parsed
            break

        answer = str((final or {}).get('answer', '')).strip()
        cited = [int(p) for p in (final or {}).get('pages', []) if isinstance(p, (int, float)) and int(p) in pages]
        abstain = (not answer) or ('KHONG_DU_BANG_CHUNG' in answer) or not cited
        traces.append({
            'query_id': item['id'], 'group': item['group'],
            'status': 'abstain' if abstain else 'answered',
            'answer': 'Không đủ bằng chứng để trả lời.' if abstain else answer,
            'citations': [f'page:{p}' for p in cited], 'rounds': rounds,
            'events': [{'event': 'search', 'query': item['question'],
                        'results': [{'id': None, 'page': p} for p in pages]}] + events,
            'elapsed_ms': (time.perf_counter() - started) * 1000,
        })
        print(item['id'], traces[-1]['status'], 'rounds', rounds, 'pages', len(pages), flush=True)

    out = HERE / 'results' / f'{args.split}-{args.tag}'
    out.mkdir(parents=True, exist_ok=True)
    config = {'split': args.split, 'mode': 'agent_vision_multiround', 'max_rounds': MAX_ROUNDS, 'max_pages': MAX_PAGES,
              'per_query': PER_QUERY, 'model': env['MODEL'], 'n': len(traces),
              'wall_seconds': round(time.perf_counter() - wall, 1), 'usage': usage}
    (out / 'traces.json').write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(config, ensure_ascii=False))


if __name__ == '__main__':
    main()
