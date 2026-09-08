"""Build six GOST-style DOCX reports from the retained ITMO reference."""
from pathlib import Path
from copy import deepcopy
import re,sys,json,hashlib
from docx import Document
from docx.shared import Cm,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'Отчет_по_производственной_практике_PrezGen_2 (2).docx'
if not REFERENCE.exists(): REFERENCE=ROOT/'assignment/report_template.docx'
TITLES={1:'Один текст — разные представления',2:'Разметка неоднозначной категории',3:'Классика против LLM',4:'Промпт как программа',5:'Мини-RAG с доказательствами',6:'AI-детектив'}

def clean(text):
    text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'\1 (\2)',text)
    return text.replace('**','').replace('`','').strip()

def font(run,size=14,bold=None):
    run.font.name='Times New Roman';run.font.size=Pt(size);run.font.color.rgb=RGBColor(0,0,0)
    run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Times New Roman')
    if bold is not None:run.bold=bold

def paragraph(doc,text='',style=None,align=None,indent=True):
    p=doc.add_paragraph(style=style);p.paragraph_format.first_line_indent=Cm(1.25 if indent else 0)
    p.paragraph_format.space_after=Pt(0);p.paragraph_format.space_before=Pt(0);p.paragraph_format.line_spacing=1.5
    p.alignment=align if align is not None else WD_ALIGN_PARAGRAPH.JUSTIFY
    font(p.add_run(clean(text)))
    return p

def heading(doc,title,level=1,newpage=False,center=False):
    p=paragraph(doc,title,f'Heading {level}',WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT,indent=not center)
    p.paragraph_format.page_break_before=newpage;p.paragraph_format.keep_with_next=True
    p.paragraph_format.space_after=Pt(12);p.paragraph_format.space_before=Pt(0 if newpage else 12)
    for r in p.runs:font(r,bold=True)
    return p

def cover(doc,num,title):
    # Clone the existing cover components rather than rebuilding the source package.
    ref=Document(REFERENCE)
    body=doc._element.body
    for child in list(body):
        if child.tag!=qn('w:sectPr'):body.remove(child)
    def clone(index,text=None):
        el=deepcopy(ref.paragraphs[index]._p);body.insert(len(body)-1,el)
        from docx.text.paragraph import Paragraph
        p=Paragraph(el,doc._body)
        if text is not None:
            p.clear();font(p.add_run(text),bold=index in [8,9])
        p.paragraph_format.first_line_indent=Cm(0)
        p.paragraph_format.line_spacing=1.0
        p.paragraph_format.space_before=Pt(0);p.paragraph_format.space_after=Pt(0)
        return p
    for i in range(7):clone(i)
    p=clone(8,'О Т Ч Е Т');p.paragraph_format.space_before=Pt(54)
    clone(9,f'по лабораторной работе № {num}')
    p=clone(9,'по дисциплине «Автоматическая обработка текста»')
    for r in p.runs:r.bold=False
    p=clone(10,f'Тема: «{title}»');p.paragraph_format.space_before=Pt(18)
    p=clone(12,'Обучающийся: Дущенко Даниил Александрович, К3341');p.paragraph_format.space_before=Pt(62)
    p=clone(14,'Преподаватель: Гусарова Наталия Федоровна');p.paragraph_format.space_before=Pt(16)
    p=clone(20,'Санкт-Петербург');p.paragraph_format.space_before=Pt(96)
    clone(21,'2026')


def table(doc,rows,number):
    rows=[[clean(c) for c in row] for row in rows]
    if len(rows)<2:return
    n=len(rows[0]);rows=[r for r in rows if len(r)==n]
    cap=paragraph(doc,f'Таблица {number} — Результаты сравнения',indent=False,align=WD_ALIGN_PARAGRAPH.LEFT)
    cap.paragraph_format.keep_with_next=True
    tab=doc.add_table(rows=1,cols=n);tab.style='Table Grid';tab.autofit=False
    first=0.32 if n<=5 else 0.22
    widths=[16.5*first]+[16.5*(1-first)/(n-1)]*(n-1) if n>1 else [16.5]
    tblpr=tab._tbl.tblPr
    tw=tblpr.find(qn('w:tblW'));tw.set(qn('w:type'),'dxa');tw.set(qn('w:w'),str(round(16.5/2.54*1440)))
    ind=OxmlElement('w:tblInd');ind.set(qn('w:w'),'0');ind.set(qn('w:type'),'dxa');tblpr.append(ind)
    for c,w in zip(tab.columns,widths):c.width=Cm(w)
    for ri,row in enumerate(rows):
        cells=tab.rows[0].cells if ri==0 else tab.add_row().cells
        for cell,text,w in zip(cells,row,widths):
            cell.width=Cm(w);p=cell.paragraphs[0];p.paragraph_format.first_line_indent=Cm(0);p.paragraph_format.line_spacing=1.0;p.paragraph_format.space_after=Pt(3);p.paragraph_format.space_before=Pt(3)
            # Keep compact tables whole; long tables retain repeating headers.
            p.paragraph_format.keep_with_next = ri == 0 or (len(rows) <= 10 and ri < len(rows)-1)
            font(p.add_run(text),size=12 if n<=5 else 10,bold=ri==0)
        if ri==0:
            rep=OxmlElement('w:tblHeader');tab.rows[0]._tr.get_or_add_trPr().append(rep)
        no=OxmlElement('w:cantSplit');tab.rows[ri]._tr.get_or_add_trPr().append(no)
    paragraph(doc,'',indent=False).paragraph_format.line_spacing=1.0


def add_contents(doc):
    h=heading(doc,'СОДЕРЖАНИЕ',newpage=True,center=True); h.style=doc.styles['Normal']; h.runs[0].bold=True
    p=paragraph(doc,'',indent=False)
    r=p.add_run();begin=OxmlElement('w:fldChar');begin.set(qn('w:fldCharType'),'begin');r._r.append(begin)
    instruction=OxmlElement('w:instrText');instruction.set(qn('xml:space'),'preserve');instruction.text=' TOC \\o "1-2" \\h \\z \\u ';r._r.append(instruction)
    sep=OxmlElement('w:fldChar');sep.set(qn('w:fldCharType'),'separate');r._r.append(sep)
    t=OxmlElement('w:t');t.text='Содержание обновляется при открытии в Word.';r._r.append(t)
    end=OxmlElement('w:fldChar');end.set(qn('w:fldCharType'),'end');r._r.append(end)


def build(folder):
    num=int(folder.name[3:5]);src=folder/'report.md'
    if not src.exists():return
    title=TITLES[num];doc=Document(REFERENCE)
    sec=doc.sections[0];sec.page_width=Cm(21);sec.page_height=Cm(29.7);sec.left_margin=Cm(3);sec.right_margin=Cm(1.5);sec.top_margin=Cm(2);sec.bottom_margin=Cm(2);sec.different_first_page_header_footer=True
    for name in ['Normal','Heading 1','Heading 2']:
        s=doc.styles[name];s.font.name='Times New Roman';s.font.size=Pt(14);s.font.color.rgb=RGBColor(0,0,0)
    doc.core_properties.author='Дущенко Даниил Александрович';doc.core_properties.title=f'Лабораторная работа № {num}. {title}';doc.core_properties.subject='Автоматическая обработка текста';doc.core_properties.comments='Подготовлено с помощью ИИ; результаты и ограничения описаны в отчете.'
    cover(doc,num,title);add_contents(doc)
    lines=src.read_text(encoding='utf-8-sig').splitlines();i=0;sub=0;main=False;intro=False;tbl=0;fig=0;in_code=False
    while i<len(lines):
        line=lines[i].strip();i+=1
        if not line or line.startswith('# '):continue
        if line.startswith('Дисциплина:') or line.startswith('Дущенко Даниил'):continue
        if line.startswith('```'):
            in_code=not in_code;continue
        if in_code:
            p=paragraph(doc,line,indent=False,align=WD_ALIGN_PARAGRAPH.LEFT)
            for r in p.runs:font(r,11)
            continue
        if line.startswith('##'):
            text=line.lstrip('#').strip();text=re.sub(r'^\d+[.)]?\s+','',text)
            if not intro:
                heading(doc,'ВВЕДЕНИЕ',newpage=True,center=True);intro=True
            elif re.search(r'источник|литератур',text,re.I):heading(doc,'СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ',newpage=True,center=True)
            elif text.lower() in ['заключение','вывод','выводы','итоговые выводы']:heading(doc,'ЗАКЛЮЧЕНИЕ',newpage=True,center=True)
            else:
                if not main:heading(doc,'1 Выполнение работы',newpage=True);main=True
                sub+=1;heading(doc,f'1.{sub} {text}',level=2)
            continue
        if line.startswith('|'):
            rows=[]
            while True:
                vals=[x.strip() for x in line.strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+',v) for v in vals):rows.append(vals)
                if i>=len(lines) or not lines[i].strip().startswith('|'):break
                line=lines[i].strip();i+=1
            tbl+=1;table(doc,rows,tbl);continue
        match=re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)',line)
        if match:
            img=folder/match.group(2)
            if img.exists():
                p=paragraph(doc,'',indent=False,align=WD_ALIGN_PARAGRAPH.CENTER);p.paragraph_format.keep_with_next=True;p.add_run().add_picture(str(img),width=Cm(16.2))
                fig+=1;paragraph(doc,f'Рисунок {fig} — {match.group(1)}',indent=False,align=WD_ALIGN_PARAGRAPH.CENTER)
            continue
        if line.startswith('- '):
            paragraph(doc,line[2:],style='List Bullet',indent=False);continue
        paragraph(doc,line)
    settings=doc.settings.element
    upd=settings.find(qn('w:updateFields'))
    if upd is None:upd=OxmlElement('w:updateFields');settings.append(upd)
    upd.set(qn('w:val'),'false')
    target=folder/'report.docx';doc.save(target);print(target)


def guide():
    doc=Document(REFERENCE);body=doc._element.body
    for child in list(body):
        if child.tag!=qn('w:sectPr'):body.remove(child)
    doc.sections[0].different_first_page_header_footer=False
    # A separate 1–2-page working instruction: source font family, compact12pt allowed by GOST.
    for line in (ROOT/'lab02_annotation/guidelines.md').read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():continue
        if line.startswith('#'):p=heading(doc,line.lstrip('# '),level=2)
        else:p=paragraph(doc,line)
        p.paragraph_format.line_spacing=1.0;p.paragraph_format.space_after=Pt(4)
        for r in p.runs:font(r,12,bold=line.startswith('#'))
    doc.core_properties.title='Инструкция разметчику: срочность обращения';doc.core_properties.author='Дущенко Даниил Александрович'
    doc.save(ROOT/'lab02_annotation/guidelines.docx')

if __name__=='__main__':
    for folder in sorted(ROOT.glob('lab0*')):
        if len(sys.argv)==1 or str(int(folder.name[3:5])) in sys.argv[1:]:build(folder)
    if len(sys.argv)==1 or '2' in sys.argv[1:]:guide()

