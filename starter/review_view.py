"""In từng câu của trace agent để người làm đọc và tự chấm. Chỉ đọc, không ghi file, không chấm thay người.

  python3 starter/review_view.py --traces starter/results/agent-dev/traces.json            # tất cả câu
  python3 starter/review_view.py --traces starter/results/agent-dev/traces.json --id q02  # một câu
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'converted'


def wrap(text, width=110, indent='    '):
    lines = []
    for para in str(text).split('\n'):
        while len(para) > width:
            cut = para.rfind(' ', 0, width)
            cut = cut if cut > 0 else width
            lines.append(indent + para[:cut])
            para = para[cut:].lstrip()
        lines.append(indent + para)
    return '\n'.join(lines)


def show(trace, questions, chunks):
    qid = trace['query_id']
    q = questions.get(qid)
    bar = '=' * 110
    print(bar)
    print(f"[{qid}] trạng thái: {trace.get('status')}   loại: {'thiếu evidence' if q is None or q.get('answerable') is False else 'có đáp án'}")
    if trace.get('status') == 'error':
        print('  LỖI:', trace.get('error'))
        return
    if q is None:
        # Câu thiếu evidence không nằm trong questions.json; lấy câu hỏi từ cases.json.
        q = CASES.get(qid, {})
    print('CÂU HỎI:')
    print(wrap(q.get('question', '')))
    if 'expected_answer' in q:
        print('ĐÁP ÁN MONG ĐỢI:')
        print(wrap(q['expected_answer']))
        for g in q.get('gold', []):
            print(f"EVIDENCE GỐC (trang {g['page']}):")
            print(wrap(g['evidence']))
    else:
        print('ĐÁP ÁN MONG ĐỢI: không có (câu thiếu evidence; chỉ chấm có từ chối đúng hay không)')
    print('\nTRẢ LỜI CỦA AGENT:')
    print(wrap(trace.get('answer', '')))
    print('CITATIONS (theo thứ tự):', trace.get('citations'))
    for cid in trace.get('citations', []):
        chunk = chunks.get(cid)
        print(f'  - {cid}')
        print(wrap(chunk['text'] if chunk else '(không có trong chunks.json)', indent='      '))
    print('\nCÁC LƯỢT SEARCH:')
    n = 0
    for e in trace.get('events', []):
        if e.get('event') == 'search':
            n += 1
            print(f"  {n}. query: {e['query']!r}   filter: {e.get('filter')}")
            print(f"     top-3: {[r['id'] for r in e['results']]}")
            print(f"     đưa vào context: {e['context_ids']}")
    if n == 0:
        print('  (không có lượt search)')
    print('\nKẾT THÚC:', [e['event'] for e in trace.get('events', []) if e.get('event') not in ('search',)])


CASES = {}


def main():
    parser = argparse.ArgumentParser(description='In từng câu của trace agent để tự chấm')
    parser.add_argument('--traces', required=True, type=Path)
    parser.add_argument('--id', help='Chỉ in một câu, ví dụ q02 hoặc u01')
    args = parser.parse_args()

    traces = json.loads(args.traces.read_text(encoding='utf-8'))
    questions = {q['id']: q for q in json.loads((DATA / 'questions.json').read_text(encoding='utf-8'))}
    CASES.update({c['id']: c for c in json.loads((DATA / 'cases.json').read_text(encoding='utf-8'))})
    chunks = {c['id']: c for c in json.loads((DATA / 'chunks.json').read_text(encoding='utf-8'))}

    if args.id:
        traces = [t for t in traces if t['query_id'] == args.id]
        if not traces:
            raise SystemExit(f'không có câu {args.id} trong trace')
    for trace in traces:
        show(trace, questions, chunks)
    print('=' * 110)
    print(f'Tổng số câu: {len(traces)}')


if __name__ == '__main__':
    main()
