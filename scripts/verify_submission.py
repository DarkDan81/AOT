"""Read-only completeness audit. No LLM calls; writes one JSON audit summary.

Run: python scripts/verify_submission.py [--output output/verification.json]
Exit 0 means all checks passed; exit 1 means missing, incomplete or invalid artifacts.
Report word limits count the Markdown main body (first section through conclusion),
excluding the title block and bibliography; Markdown markup is not counted as words.
"""
import argparse
import csv
import itertools
import json
import math
import re
import sys
import zipfile
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FOLDERS={1:'lab01_representations',2:'lab02_annotation',3:'lab03_classification',4:'lab04_prompt_program',5:'lab05_mini_rag',6:'lab06_ai_detective'}
CHECKS=[]
def csvrows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def jsonrows(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def js(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def require(condition,detail):
    if not condition:raise AssertionError(detail)
def check(lab,name,fn):
    try:
        detail=fn()
        CHECKS.append({'lab':lab,'check':name,'status':'passed','detail':detail})
    except FileNotFoundError as e:
        CHECKS.append({'lab':lab,'check':name,'status':'missing','detail':str(e)})
    except Exception as e:
        CHECKS.append({'lab':lab,'check':name,'status':'failed','detail':f'{type(e).__name__}: {e}'})
def count_csv(p,n,unique='id'):
    rows=csvrows(p);require(len(rows)==n,f'{p.name}: expected {n}, observed {len(rows)}')
    require(len({r[unique] for r in rows})==n,f'{p.name}: duplicate IDs')
    return {'rows':n,'unique_ids':n}
def lab1_data(p):
    texts=csvrows(p/'data/texts.csv');tr=csvrows(p/'data/transformations.csv');sim=csvrows(p/'results/similarities.csv')
    require(len(texts)==150 and len({x['id'] for x in texts})==150,'Expected 150 unique source texts')
    require(len(tr)==len(sim)==9,'Expected nine transformations and nine similarity rows')
    require(len({r['label'] for r in tr})==3,'Three transformed source classes required')
    require(Counter(r['id'] for r in tr)==dict.fromkeys({r['id'] for r in tr},3) and len({r['id'] for r in tr})==3,'Three source texts with three variants each required')
    return {'texts':150,'transformations':9,'classes':sorted({r['label'] for r in tr})}
def lab1_notebook(p):
    nb=js(p/'lab01.ipynb');cells=[c for c in nb['cells'] if c['cell_type']=='code' and ''.join(c.get('source',[])).strip()]
    require(cells and all(c.get('execution_count') is not None for c in cells),'Unexecuted code cell(s)')
    require(not any(o.get('output_type')=='error' for c in cells for o in c.get('outputs',[])),'Notebook contains error output')
    return {'executed_code_cells':len(cells)}
def agreement(rows):
    cols=['annotator_'+a for a in 'abc'];n=len(rows);labels=['0','1','?'];out={}
    for a,b in itertools.combinations(cols,2):
        ca=Counter(r[a] for r in rows);cb=Counter(r[b] for r in rows)
        observed=sum(r[a]==r[b] for r in rows)/n;expected=sum(ca[l]*cb[l] for l in labels)/n**2
        out[a+' / '+b]=(observed-expected)/(1-expected)
    total=Counter(r[c] for r in rows for c in cols)
    observed=sum(sum(v*(v-1) for v in Counter(r[c] for c in cols).values())/6 for r in rows)/n
    expected=sum((total[l]/(3*n))**2 for l in labels)
    out['Fleiss kappa']=(observed-expected)/(1-expected)
    out['full agreement']=sum(len({r[c] for c in cols})==1 for r in rows)/n
    return out
def lab2(p):
    texts=csvrows(p/'data/texts.csv');selected=csvrows(p/'data/disputed30.csv');ids={r['id'] for r in texts};same={r['id'] for r in selected}
    require(len(texts)==len(ids)==90 and len(selected)==len(same)==30 and same<=ids,'Expected 90 texts and 30 unique selected IDs')
    initial={i:{'id':i} for i in ids};revised={i:{'id':i} for i in same}
    for a in 'abc':
        for stage,target,expected in [('initial',initial,ids),('revised',revised,same)]:
            rows=csvrows(p/f'annotations/{stage}_{a}.csv')
            require(len(rows)==len(expected) and {r['id'] for r in rows}==expected,f'{stage}_{a}: ID coverage mismatch')
            require(all(r['label'] in ['0','1','?'] for r in rows),f'{stage}_{a}: invalid label')
            for r in rows:target[r['id']]['annotator_'+a]=r['label']
    frames={'initial_90':list(initial.values()),'initial_disputed_30':[initial[i] for i in same],'revised_same_30':list(revised.values())}
    metrics=csvrows(p/'results/agreement_metrics.csv');require(len(metrics)==15,'Expected 15 agreement metric rows')
    expected={(stage,name):(value,len(rows)) for stage,rows in frames.items() for name,value in agreement(rows).items()}
    require({(r['stage'],r['metric']) for r in metrics}==set(expected),'Missing/duplicate stages or metrics')
    for r in metrics:
        value,n=expected[(r['stage'],r['metric'])]
        require(int(r['n'])==n and math.isclose(float(r['value']),value,abs_tol=1e-10),'Agreement metrics do not recompute on same 30 IDs')
    return {'initial':90,'revised_same_ids':30,'annotators':3,'recomputed_metrics':15}
def lab3_responses(p):
    rows=jsonrows(p/'results/llm_responses.jsonl');ids=set()
    for split,file in [('test','test'),('stress','stress')]:ids|={(r['id'],split) for r in csvrows(p/f'data/{file}.csv')}
    expected={(i,s,m) for i,s in ids for m in ['zero_shot','few_shot']}
    actual={(r['id'],r['set'],r['mode']) for r in rows}
    require(len(rows)==1200 and len(actual)==1200 and actual==expected,f'Expected 1200 unique LLM calls; observed rows={len(rows)}, unique={len(actual)}')
    require(all('content' in r and 'usage' in r and 'model' in r for r in rows),'Missing actual response provenance')
    return {'responses':1200,'valid':sum(bool(r.get('valid')) for r in rows)}
def lab3_predictions(p):
    rows=csvrows(p/'results/predictions.csv');require(len(rows)==600 and len({r['id'] for r in rows})==600,'Expected 600 unique final predictions')
    require(Counter(r['set'] for r in rows)=={'test':500,'stress':100},'Expected test500/stress100')
    methods=['majority','tfidf_lr','zero_shot','few_shot']
    require(all(all(r.get(m) for m in methods) for r in rows),'Missing prediction for one of four methods')
    require(all(r[m] in ['positive','negative','neutral'] for r in rows for m in ['majority','tfidf_lr']),'Invalid classical label')
    return {'rows':600,'methods':methods}
def lab4(p):
    data=jsonrows(p/'data/reviews.jsonl');rows=jsonrows(p/'results/predictions.jsonl')
    ids={r['id'] for r in data};require(len(data)==len(ids)==50,'Expected 50 unique reviews')
    expected={(i,v) for i in ids for v in ['version_1','version_2']}
    require(len(rows)==100 and {(r['id'],r['version']) for r in rows}==expected,'Expected 50x2 predictions')
    for r in rows:
        a=r['attempts'];require(1<=len(a)<=2,'More than one retry or absent request')
        require((len(a)==2)==(a[0].get('error') is not None),'Retry must occur exactly once iff first contract failed')
    return {'reviews':50,'predictions':100,'retries':sum(len(r['attempts'])-1 for r in rows)}
def lab5(p):
    docs=jsonrows(p/'data/documents.jsonl');questions=jsonrows(p/'data/questions.jsonl');gold=jsonrows(p/'data/train.jsonl')+jsonrows(p/'evaluation_private/test_gold.jsonl');rows=jsonrows(p/'results/predictions.jsonl')
    require(len(docs)==30 and len({d['document_id'] for d in docs})==30,'Expected 30 unique documents')
    require(all(20<=len(d['paragraphs'])<=50 and len({x['id'] for x in d['paragraphs']})==len(d['paragraphs']) for d in docs),'Paragraph volume or stable IDs invalid')
    ids={q['question_id'] for q in questions};require(len(questions)==len(ids)==100 and len(gold)==100,'Expected 100 questions and golds')
    require({q['question_id'] for q in gold}==ids,'Question/gold coverage mismatch')
    unanswerable=sum(not q['answerable'] for q in gold);require(unanswerable>=20,'Fewer than 20 questions without answers')
    require(len(rows)==200 and {(r['question_id'],r['mode']) for r in rows}=={(i,m) for i in ids for m in ['full','bm25']},'Expected 200 main predictions')
    return {'documents':30,'paragraphs_min':min(len(d['paragraphs']) for d in docs),'paragraphs_max':max(len(d['paragraphs']) for d in docs),'questions':100,'unanswerable':unanswerable,'main_responses':200}
def lab6_data(p):
    docs=jsonrows(p/'data/documents.jsonl');claims=jsonrows(p/'data/claims.jsonl');questions=jsonrows(p/'data/questions.jsonl')
    word_count=sum(len(d['text'].split()) for d in docs)
    require(8<=len(docs)<=10 and 5000<=word_count<=8000,f'Documents={len(docs)}, words={word_count}; expected 8–10 and 5000–8000')
    require(30<=len(claims)<=40 and len({c['claim_id'] for c in claims})==len(claims),'Expected 30–40 unique claims')
    lookup={d['document_id']:d for d in docs};require(all(c['document_id'] in lookup and c['evidence'] in lookup[c['document_id']]['text'] for c in claims),'Claim source or verbatim evidence invalid')
    counts=Counter(q['type'] for q in questions);types=['direct','multi_hop','temporal','rule_exception','contradiction','insufficient']
    require(len(questions)==12 and len({q['question_id'] for q in questions})==12 and counts==dict.fromkeys(types,2),f'Expected two questions of each six types: {dict(counts)}')
    return {'documents':len(docs),'words':word_count,'claims':len(claims),'questions':12,'types':dict(counts)}
def lab6_responses(p):
    variants=[v for v in jsonrows(p/'data/variants.jsonl') if v['applicable']];rows=jsonrows(p/'results/responses.jsonl')
    expected={(v['variant_id'],m) for v in variants for m in ['A','B','C']};actual={(r['variant_id'],r['mode']) for r in rows}
    require(len(expected)==120,'Expected exactly 40 applicable variants x3')
    require(len(rows)==120 and actual==expected,f'Expected all 120 responses, observed {len(rows)}, unique={len(actual)}')
    lookup={(r['variant_id'],r['mode']):r for r in rows}
    for v in variants:
        b=lookup[(v['variant_id'],'B')];c=lookup[(v['variant_id'],'C')]
        require(set(b['context_claim_ids'])==set(c['context_claim_ids']),f"Different B/C claim sets: {v['variant_id']}")
        require(len(b['context_claim_ids'])<=15,'B retrieval exceeds 15')
    return {'responses':120,'valid':sum(bool(r['valid']) for r in rows),'identical_bc_pairs':len(variants)}
def reports(p):
    docx=p/'report.docx';pdf=p/'report.pdf'
    require(docx.is_file() and pdf.is_file(),f'Missing report.docx or report.pdf in {p.name}')
    with zipfile.ZipFile(docx) as z:require('word/document.xml' in z.namelist(),'Invalid DOCX structure')
    require(pdf.read_bytes().startswith(b'%PDF-'),'Invalid PDF header')
    return {'docx_bytes':docx.stat().st_size,'pdf_bytes':pdf.stat().st_size}
def words(p,lower,upper):
    text=(p/'report.md').read_text(encoding='utf-8-sig');lines=text.splitlines();start=next((i for i,l in enumerate(lines) if l.startswith('## ')),0)
    selected=[]
    for l in lines[start:]:
        if l.startswith('## ') and any(s in l.lower() for s in ['источник','литератур','библиограф']):break
        if re.match(r'^\s*\|?\s*:?-{3,}',l):continue
        l=re.sub(r'!\[[^]]*\]\([^)]*\)','',l)
        l=re.sub(r'\[([^]]+)\]\([^)]*\)',r'\1',l)
        selected.extend(t for t in l.split() if re.search(r'\w',t))
    n=len(selected);require(lower<=n<=upper,f'Main-body word count={n}, required {lower}..{upper}')
    return {'main_body_words':n,'minimum':lower,'maximum':upper,'method':'whitespace tokens containing letters/digits; title block and bibliography excluded'}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='output/submission_verification.json');args=parser.parse_args()
    p={i:ROOT/f for i,f in FOLDERS.items()}
    check(1,'150_texts_9_variants_3_classes',lambda:lab1_data(p[1]));check(1,'executed_notebook',lambda:lab1_notebook(p[1]))
    check(2,'90_initial_30_revised_triple_labels_same30_metrics',lambda:lab2(p[2]))
    for file,n in [('train',3000),('validation',500),('test',500),('stress',100)]:check(3,'data_'+file,lambda file=file,n=n:count_csv(p[3]/f'data/{file}.csv',n))
    check(3,'1200_actual_llm_responses',lambda:lab3_responses(p[3]));check(3,'600_predictions_four_methods',lambda:lab3_predictions(p[3]))
    check(4,'50_reviews_two_versions_exactly_one_conditional_retry',lambda:lab4(p[4]))
    check(5,'30_documents_100_questions_200_main_responses',lambda:lab5(p[5]))
    check(6,'documents_claims_questions',lambda:lab6_data(p[6]));check(6,'120_responses_and_equal_bc_contexts',lambda:lab6_responses(p[6]))
    for i in range(1,7):check(i,'docx_and_pdf',lambda i=i:reports(p[i]))
    for i,lo,hi in [(1,700,1000),(2,0,1200),(6,0,1500)]:check(i,'report_word_limit',lambda i=i,lo=lo,hi=hi:words(p[i],lo,hi))
    failed=[c for c in CHECKS if c['status']!='passed']
    summary={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'root':str(ROOT),'complete':not failed,'checks_passed':len(CHECKS)-len(failed),'checks_total':len(CHECKS),'pending_labs':sorted({c['lab'] for c in failed}),'checks':CHECKS}
    out=Path(args.output);out=out if out.is_absolute() else ROOT/out;out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return 1 if failed else 0
if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf8')
    raise SystemExit(main())
