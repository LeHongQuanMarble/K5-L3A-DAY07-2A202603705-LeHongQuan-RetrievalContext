"""Xây index cho thử thách nâng cao: Annual Energy Review 2011 (EIA, 390 trang, phạm vi công cộng Hoa Kỳ).

Pipeline:
1. Text theo trang bằng `pdftotext -layout` (giữ bố cục cột; bảng vẫn là text có hàng/cột).
2. Chunk theo đoạn, tối đa ~900 ký tự, mỗi chunk gắn số trang PDF và loại (text/figure).
3. Trang có tiêu đề "Figure N": render PNG, nhờ Claude (vision) mô tả hình; mô tả là dữ liệu do model tạo,
   được lưu kèm đường dẫn ảnh để kiểm chứng. Kết quả cache trong figures.json.
4. Embedding (MiniLM đa ngôn ngữ, cùng model Day 6) cho toàn bộ chunk, lưu vectors.npy.

Chạy bằng venv của Day 6 (có sentence-transformers và anthropic):
  <day6>/starter/.venv/bin/python starter/advanced/build_index.py --steps text,figures,embed
"""
import argparse
import base64
import concurrent.futures
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]  # starter/
DATA = ROOT / 'data' / 'advanced'
PDF = DATA / 'source' / 'aer_2011.pdf'
DOC = 'aer2011'
MAX_CHARS = 900
FIGURE_RE = re.compile(r'\bFigure\s+\d+')
EMBED_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'


def page_texts(pdf):
    out = subprocess.run(['pdftotext', '-layout', str(pdf), '-'], capture_output=True, text=True, check=True).stdout
    pages = out.split('\f')
    if pages and not pages[-1].strip():
        pages = pages[:-1]
    return [clean(p) for p in pages]


def clean(page):
    lines = []
    for line in page.splitlines():
        line = re.sub(r'(?:\s?\.){4,}', ' ...', line)  # leader dots của mục lục
        line = re.sub(r'\s{3,}', ' | ', line.strip())  # cột bảng: giữ dấu phân cách
        lines.append(line.rstrip())
    return '\n'.join(lines).strip()


YEAR_ROW = re.compile(r'^\s*(?:19|20)\d\d\b')
CAPTION = re.compile(r'^\s*(Table|Figure)\s+\d+(\.\d+)?\s')


def page_header(text):
    """Tiêu đề bảng/hình và dòng tên cột của trang; gắn vào mọi chunk để chunk sau không mất ngữ cảnh."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    caption_idx = next((i for i, l in enumerate(lines) if CAPTION.match(l)), None)
    if caption_idx is None:
        return lines[0][:200] if lines else ''
    head = [lines[caption_idx]]
    for l in lines[caption_idx + 1:]:
        if YEAR_ROW.match(l) or len(head) >= 8:
            break
        head.append(l)
    return ' / '.join(head)[:350]


def chunk_page(doc, page_no, text):
    header = page_header(text)
    if header and re.search(r'^\s*(?:19|20)\d\d\b', text, re.M):  # bảng theo năm: mỗi hàng một chunk, kèm tiêu đề và tên cột
        out, k = [], 0
        for line in text.splitlines():
            if YEAR_ROW.match(line):
                out.append({'id': f'{doc}:p{page_no:03d}:r{k:03d}', 'text': f'[{header}]\n{line.strip()}', 'source_id': doc,
                            'source': PDF.name, 'page': page_no, 'type': 'text_row',
                            'metadata': {'has_figure_caption': bool(FIGURE_RE.search(text))}})
                k += 1
        if out:
            return out
    limit = MAX_CHARS - (len(header) + 2 if header else 0)
    paras, cur = [], []
    for line in text.splitlines():
        if not line.strip():
            if cur:
                paras.append('\n'.join(cur))
                cur = []
        else:
            cur.append(line)
    if cur:
        paras.append('\n'.join(cur))
    chunks, buf = [], ''
    for para in paras:
        pieces = [para] if len(para) <= limit else [para[i:i + limit] for i in range(0, len(para), limit)]
        for piece in pieces:
            if buf and len(buf) + len(piece) + 2 > limit:
                chunks.append(buf)
                buf = piece
            else:
                buf = f'{buf}\n\n{piece}' if buf else piece
    if buf:
        chunks.append(buf)
    out = []
    for k, body in enumerate(chunks):
        body = f'[{header}]\n{body}' if header else body
        out.append({'id': f'{doc}:p{page_no:03d}:t{k:02d}', 'text': body, 'source_id': doc, 'source': PDF.name,
                    'page': page_no, 'type': 'text', 'metadata': {'has_figure_caption': bool(FIGURE_RE.search(text))}})
    return out


def render_page(page_no):
    out_prefix = DATA / 'pages' / f'p{page_no:03d}'
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['pdftoppm', '-f', str(page_no), '-l', str(page_no), '-r', '110', '-png', '-singlefile',
                    str(PDF), str(out_prefix)], check=True)
    return out_prefix.with_suffix('.png')


FIGURE_PROMPT = (
    'Trang này từ Annual Energy Review 2011 của EIA. Mô tả từng hình/biểu đồ trên trang để người đọc có thể tra cứu: '
    'số hình và tiêu đề; đơn vị; tên trục và nhãn; các chuỗi/nhóm; khoảng năm hoặc danh mục; giá trị đọc được '
    '(nếu không có nhãn số thì ghi là đọc gần đúng); xu hướng chính. Chỉ dùng thông tin nhìn thấy trên trang. '
    'Nếu không đọc được, ghi "không đọc được". Viết bằng tiếng Anh, ngắn gọn.\n\nText trích từ trang:\n{text}'
)


def describe_figures(pending, model, client):
    def one(page_no):
        png = render_page(page_no)
        data = base64.b64encode(png.read_bytes()).decode()
        prompt = FIGURE_PROMPT.format(text=pages[page_no - 1][:2500])
        resp = client.messages.create(
            model=model, max_tokens=3000,
            messages=[{'role': 'user', 'content': [
                {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': data}},
                {'type': 'text', 'text': prompt}]}])
        text = ''.join(b.text for b in resp.content if b.type == 'text').strip()
        if not text:
            raise RuntimeError(f'page {page_no}: no text block in response')
        return page_no, text, {'in': resp.usage.input_tokens, 'out': resp.usage.output_tokens}

    results, usage = {}, {'in': 0, 'out': 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for page_no, text, u in pool.map(one, pending):
            results[str(page_no)] = {'page': page_no, 'description': text, 'image': str(render_page(page_no).relative_to(ROOT))}
            usage['in'] += u['in']
            usage['out'] += u['out']
    return results, usage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', default='text,figures,embed')
    args = parser.parse_args()
    steps = set(args.steps.split(','))
    (DATA / 'index').mkdir(parents=True, exist_ok=True)
    global pages
    pages = page_texts(PDF)
    manifest_path = DATA / 'index' / 'manifest.json'

    if 'text' in steps:
        chunks = []
        for i, text in enumerate(pages, start=1):
            if text:
                chunks.extend(chunk_page(DOC, i, text))
        figures_pages = [i for i, t in enumerate(pages, start=1) if FIGURE_RE.search(t)]
        (DATA / 'index' / 'text_chunks.json').write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding='utf-8')
        manifest = {'pdf': str(PDF.relative_to(ROOT)), 'sha256': hashlib.sha256(PDF.read_bytes()).hexdigest(),
                    'pages': len(pages), 'text_chunks': len(chunks), 'figure_pages': figures_pages,
                    'empty_pages': [i for i, t in enumerate(pages, start=1) if not t],
                    'extraction': 'pdftotext -layout; columns collapsed to " | "; chunks <= %d chars' % MAX_CHARS}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({k: manifest[k] for k in ('pages', 'text_chunks')}), 'figure pages:', len(figures_pages))

    if 'figures' in steps:
        cache_path = DATA / 'index' / 'figures.json'
        cache = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() else {}
        figure_pages = json.loads(manifest_path.read_text(encoding='utf-8'))['figure_pages']
        pending = [p for p in figure_pages if str(p) not in cache]
        from anthropic import Anthropic
        env = dotenv_values(ROOT / '.env')
        client = Anthropic(api_key=env['ANTHROPIC_API_KEY'])
        new, usage = describe_figures(pending, env['MODEL'], client)
        cache.update(new)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding='utf-8')
        print('figure descriptions:', len(cache), 'new:', len(new), 'usage:', usage)

    if 'embed' in steps:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        chunks = json.loads((DATA / 'index' / 'text_chunks.json').read_text(encoding='utf-8'))
        cache_path = DATA / 'index' / 'figures.json'
        for page_no, fig in sorted(json.loads(cache_path.read_text(encoding='utf-8')).items(), key=lambda kv: int(kv[0])):
            chunks.append({'id': f'{DOC}:p{int(page_no):03d}:f00', 'text': f'[Figure description, page {page_no}]\n{fig["description"]}',
                           'source_id': DOC, 'source': PDF.name, 'page': int(page_no), 'type': 'figure',
                           'metadata': {'image': fig['image'], 'generated_by': 'claude vision (unverified)'}})
        model = SentenceTransformer(EMBED_MODEL)
        model.max_seq_length = 256
        vectors = model.encode([c['text'] for c in chunks], normalize_embeddings=True, batch_size=32,
                               show_progress_bar=False).astype('float32')
        np.save(DATA / 'index' / 'vectors.npy', vectors)
        (DATA / 'index' / 'chunks.json').write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding='utf-8')
        print('embedded', len(chunks), 'chunks, dim', vectors.shape[1])


if __name__ == '__main__':
    main()
