"""Tổng hợp kết quả benchmark nâng cao.

Hai phần:
1. show: in từng câu (câu hỏi, đáp án mong đợi, trả lời, citation kèm trang, trang gold có trong context không)
   để người chấm đọc và ghi nhãn.
2. summarize: đọc reviews.json (mỗi câu: id, pass true/false, note) và tính tỷ lệ đạt theo nhóm.
   Câu lỗi hệ thống (status error) và câu không có nhãn được tính là không đạt và báo riêng.

  python3 starter/advanced/score.py show --run starter/advanced/results/dev-dense-k3-b2500
  python3 starter/advanced/score.py summarize --run starter/advanced/results/test-...
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
STARTER = HERE.parent
CHUNKS = {c['id']: c for c in json.loads((STARTER / 'data' / 'advanced' / 'index' / 'chunks.json').read_text(encoding='utf-8'))}


def load_bench(split):
    return {x['id']: x for x in json.loads((HERE / 'benchmark' / f'{split}.json').read_text(encoding='utf-8'))}


def gold_pages(item):
    return {g['page'] for g in item.get('gold', [])}


def retrieved_pages(trace):
    pages = set()
    for e in trace.get('events', []):
        if e.get('event') == 'search':
            for r in e.get('results', []):
                pages.add(r['page'])
    return pages


def show(run):
    split = json.loads((run / 'config.json').read_text(encoding='utf-8'))['split']
    bench = load_bench(split)
    for t in json.loads((run / 'traces.json').read_text(encoding='utf-8')):
        item = bench[t['query_id']]
        hit = bool(gold_pages(item) & retrieved_pages(t))
        print(f"## {t['query_id']} [{item['group']}] status={t['status']} gold_page_retrieved={hit if gold_pages(item) else 'n/a'}")
        print('  Q:', item['question'])
        print('  EXP:', item['expected_answer'])
        print('  A:', (t.get('answer') or '').replace('\n', ' ')[:500])
        print('  CIT:', t.get('citations'))
        print()


def summarize(run):
    split = json.loads((run / 'config.json').read_text(encoding='utf-8'))['split']
    bench = load_bench(split)
    traces = {t['query_id']: t for t in json.loads((run / 'traces.json').read_text(encoding='utf-8'))}
    reviews_path = run / 'reviews.json'
    reviews = {r['id']: r for r in json.loads(reviews_path.read_text(encoding='utf-8'))} if reviews_path.exists() else {}
    per_group = defaultdict(lambda: [0, 0])
    errors, unlabeled = [], []
    for qid, item in bench.items():
        g = item['group']
        per_group[g][1] += 1
        t = traces.get(qid)
        if t is None:
            unlabeled.append(qid)
            continue
        if t['status'] == 'error':
            errors.append(qid)
            continue
        r = reviews.get(qid)
        if r is None:
            unlabeled.append(qid)
            continue
        if r['pass']:
            per_group[g][0] += 1
    total_pass = sum(v[0] for v in per_group.values())
    total_n = sum(v[1] for v in per_group.values())
    out = {'split': split, 'total': f'{total_pass}/{total_n}', 'rate': round(100 * total_pass / total_n, 1),
           'groups': {k: f'{v[0]}/{v[1]}' for k, v in sorted(per_group.items())},
           'system_errors_counted_as_fail': errors, 'unlabeled': unlabeled}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=('show', 'summarize'))
    p.add_argument('--run', required=True, type=Path)
    a = p.parse_args()
    show(a.run) if a.cmd == 'show' else summarize(a.run)
