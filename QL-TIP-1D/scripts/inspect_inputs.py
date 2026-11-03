"""Read external sources without modifying originals; extract local audit text."""
from pathlib import Path
import hashlib
import json
from pypdf import PdfReader
from docx import Document

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    Path('C:/Users/26507/Downloads/QL-TIP-1D_本地Agent科研执行任务书.md'),
    Path('C:/Users/26507/Downloads/s41567-025-03087-1.pdf'),
    Path('C:/Users/26507/Downloads/s41567-024-02472-6.pdf'),
    Path('C:/Users/26507/Downloads/量子光注入金属纳米尖端的背景调查与可行性评估.docx'),
    Path('C:/Users/26507/.codex/attachments/cada1c1a-6dfe-4d07-9d8f-c2cb6e35e7ba/已粘贴的文本.txt'),
    Path('C:/Users/26507/.codex/attachments/7f0e154a-26d8-430a-bce9-282c6ccafd03/已粘贴的文本.txt'),
    Path('C:/Users/26507/Downloads/QL-TIP-1D_选题最终方案与Agent变更指令_20261103.md'),
]

def main():
    out = ROOT / 'references' / 'extracted'
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for i, p in enumerate(SOURCES):
        raw = p.read_bytes()
        record = dict(source_id=i, filename=p.name, original_path=str(p),
                      bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                      accessed_local_date='2026-11-04', original_modified=False)
        if p.suffix == '.pdf':
            reader = PdfReader(p)
            record['pages'] = len(reader.pages)
            text = '\n\n'.join(f'--- PDF PAGE {n+1} ---\n' + page.extract_text()
                                for n, page in enumerate(reader.pages))
        elif p.suffix == '.docx':
            doc = Document(p)
            # Include OMML formula text, which python-docx Paragraph.text omits.
            def para_text(p):
                return ''.join(n.text for n in p._p.iter() if n.tag.endswith('}t') and n.text)
            text = '\n'.join(para_text(x) for x in doc.paragraphs)
            text += '\n\n' + '\n\n'.join('\n'.join(' | '.join(c.text for c in row.cells)
                                                    for row in table.rows) for table in doc.tables)
        else:
            text = raw.decode('utf-8-sig')
        target = out / f'{i:02d}_{p.stem}.txt'
        target.write_text(text, encoding='utf-8')
        record['extraction_path'] = target.relative_to(ROOT).as_posix()
        record['extracted_characters'] = len(text)
        records.append(record)
        print(i, p.name, len(raw), len(text), record.get('pages', ''))
    (ROOT / 'references' / 'source_inventory.json').write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

if __name__ == '__main__':
    main()
