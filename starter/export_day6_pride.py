"""Adapter: Day 6 Pride-and-Prejudice corpus and benchmark -> Day 7 JSON contract.

Chunks come from the Day 6 chunking code (strategy B: paragraph groups, max 800 chars,
per page), so they match what Day 6 indexed. Questions and gold are copied field by field;
nothing is inferred, and evidence is never edited. Output is validated with baseline.validate().

The PDF is read with Day 6's own load_pdf (needs pypdf), so run with Day 6's environment:
  <day6>/starter/.venv/bin/python starter/export_day6_pride.py --day6 <day6>/starter --output starter/data/converted
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from baseline import validate  # noqa: E402

PARAGRAPH_MAX = 800  # Day 6 PARAGRAPH_MAX (strategy B)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def page_number(page_id: str) -> int:
    # 'pride-and-prejudice:p035' -> 35; must match the int page stored in chunks.
    return int(page_id.rsplit(':p', 1)[1])


def export(day6: Path, output: Path) -> dict:
    sys.path.insert(0, str(day6))
    from src.chunking import by_paragraph  # Day 6 code, reused unchanged
    from src.ingestion import load_pdf  # same page text the Day 6 pipeline indexed (keeps paragraph breaks)

    pages = load_pdf(day6 / 'data' / 'pride-and-prejudice.pdf', 'pride-and-prejudice', 'data/pride-and-prejudice.pdf')
    chunks = []
    for doc in pages:
        if doc.metadata['empty']:
            continue
        for c in by_paragraph(doc, PARAGRAPH_MAX):
            chunks.append({'id': c.chunk_id, 'text': c.text, 'source_id': c.document_id, 'source': c.source,
                           'page': c.page, 'metadata': {**c.metadata, 'day6_strategy': 'B'}})

    questions = []
    for q in read_jsonl(day6 / 'benchmark' / 'questions.jsonl'):
        questions.append({'id': q['id'], 'split': q['split'], 'type': q['type'], 'question': q['question'],
                          'expected_answer': q['expected_answer'],
                          'gold': [{'source_id': q['gold_source_id'], 'page': page_number(q['gold_page']),
                                    'evidence': q['evidence_quote']}],
                          'filter': q.get('filter')})

    validate(chunks, questions)  # raises if any gold evidence is not inside one chunk of the right source/page

    # Abstention items (no gold, answerable=false) are kept in a separate hand-checked file.
    unanswerable = json.loads((HERE.parent / 'data' / 'pride_unanswerable.json').read_text(encoding='utf-8'))
    cases = [{'id': q['id'], 'split': q['split'], 'answerable': True, 'question': q['question']} for q in questions]
    cases += [{'id': u['id'], 'split': u['split'], 'answerable': False, 'question': u['question'],
               'expected_behavior': u['expected_behavior'], 'absence_reason': u['absence_reason']}
              for u in unanswerable]
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('ID câu hỏi trùng giữa benchmark Day 6 và câu thiếu evidence')

    output.mkdir(parents=True, exist_ok=True)
    (output / 'chunks.json').write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'questions.json').write_text(json.dumps(questions, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    splits = {s: sum(c['split'] == s for c in cases) for s in ('dev', 'test')}
    return {'chunks': len(chunks), 'questions_answerable': len(questions), 'questions_unanswerable': len(unanswerable),
            'splits': splits, 'with_filter': sum(q['filter'] is not None for q in questions)}


def main():
    parser = argparse.ArgumentParser(description='Chuyển corpus và benchmark Day 6 (Pride and Prejudice) sang schema Day 7')
    parser.add_argument('--day6', required=True, type=Path, help='Thư mục starter/ của repo Day 6')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.day6.resolve(), args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
