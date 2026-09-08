"""Structural report preflight; supplements, but does not replace, visual review."""
from pathlib import Path
import hashlib
import json
import zipfile
import xml.etree.ElementTree as ET
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
W = '{' + NS['w'] + '}'


def check(folder):
    docx, pdf = folder / 'report.docx', folder / 'report.pdf'
    assert docx.exists() and pdf.exists(), f'Missing report: {folder.name}'
    with zipfile.ZipFile(docx) as archive:
        document = ET.fromstring(archive.read('word/document.xml'))
        styles = ET.fromstring(archive.read('word/styles.xml'))
        section = document.find('.//w:sectPr', NS)
        margins = section.find('w:pgMar', NS)
        for name, expected in [('left', 1701), ('right', 850), ('top', 1134), ('bottom', 1134)]:
            assert abs(int(margins.get(W + name)) - expected) <= 2, (folder.name, name)
        normal = next(s for s in styles.findall('w:style', NS)
                      if s.find('w:name', NS) is not None and s.find('w:name', NS).get(W + 'val') == 'Normal')
        assert normal.find('w:rPr/w:rFonts', NS).get(W + 'ascii') == 'Times New Roman'
        assert normal.find('w:rPr/w:sz', NS).get(W + 'val') == '28'
        assert section.find('w:titlePg', NS) is not None
    with pymupdf.open(pdf) as rendered:
        text = '\n'.join(page.get_text() for page in rendered)
        cover = rendered[0].get_text()
        assert 'К3341' in cover and 'Гусарова Наталия Федоровна' in cover
        assert 'Дущенко Даниил Александрович' in cover
        assert ('Коваленко Евгений Юрьевич' in cover) == (int(folder.name[3:5]) not in {1,3,4})
        if int(folder.name[3:5]) in {2,5,6}:
            assert text.count('Дущенко Даниил Александрович') >= 2
            assert text.count('Коваленко Евгений Юрьевич') >= 2
        assert all(s in text for s in ['СОДЕРЖАНИЕ', 'ВВЕДЕНИЕ', 'ЗАКЛЮЧЕНИЕ', 'СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ'])
        assert not any(s in text for s in ['Содержание обновляется', 'EMPIRICAL_', 'КХХХХХ', 'PrezGen'])
        for i, page in enumerate(rendered, 1):
            assert abs(page.rect.width - 595.28) < 1 and abs(page.rect.height - 841.89) < 1
            assert page.get_text().strip(), f'Empty page {folder.name}:{i}'
            for x0, y0, x1, y1, *_ in page.get_text('words'):
                assert 0 <= x0 < x1 <= page.rect.width and 0 <= y0 < y1 <= page.rect.height, f'Text outside page {folder.name}:{i}'
        pages = len(rendered)
    return {'folder': folder.name, 'pages': pages, 'a4': True, 'margins_cm': [3, 1.5, 2, 2],
            'normal_font': 'Times New Roman 14 pt', 'required_sections': True,
            'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [docx, pdf]}}


if __name__ == '__main__':
    records = [check(folder) for folder in sorted(ROOT.glob('lab0*')) if folder.is_dir()]
    assert len(records) == 6
    guide_files = [ROOT / 'lab02_annotation' / ('guidelines.' + ext) for ext in ['docx', 'pdf']]
    assert all(p.exists() for p in guide_files)
    with pymupdf.open(guide_files[1]) as guide:
        assert 1 <= len(guide) <= 2
        guide_pages = len(guide)
    result = {'reports': records, 'annotation_guidelines': {'pages': guide_pages,
              'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in guide_files}},
              'scope': 'Structural preflight; page images reviewed separately.'}
    (ROOT / 'output').mkdir(exist_ok=True)
    (ROOT / 'output/report_layout_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
