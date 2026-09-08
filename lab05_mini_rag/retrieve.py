import json
from pathlib import Path
from razdel import tokenize
from rank_bm25 import BM25Okapi
ROOT=Path(__file__).parent
def read(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
def tokens(text):return [x.text.lower() for x in tokenize(text) if any(c.isalnum() for c in x.text)]
def top5(document,question):
    ps=document['paragraphs']; scores=BM25Okapi([tokens(p['text']) for p in ps]).get_scores(tokens(question))
    return [ps[i] for i in sorted(range(len(ps)),key=lambda i:(-scores[i],i))[:5]]
def main():
    docs={x['document_id']:x for x in read(ROOT/'data/documents.jsonl')}
    gold=read(ROOT/'data/train.jsonl')+read(ROOT/'evaluation_private/test_gold.jsonl')
    rows=[]
    for q in gold:
        ids=[x['id'] for x in top5(docs[q['document_id']],q['question'])]; expected=set(q['evidence_ids'])
        rows.append({'question_id':q['question_id'],'split':q['split'],'top5':ids,'gold_evidence_ids':sorted(expected),'recall':len(expected.intersection(ids))/len(expected) if expected else None,'all_evidence':expected.issubset(ids) if expected else None})
    out=ROOT/'results';out.mkdir(exist_ok=True)
    (out/'retrieval_before_llm.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    for split in ['train','test']:
        vals=[r['recall'] for r in rows if r['split']==split and r['recall'] is not None]
        print(split, 'macro evidence recall@5=',sum(vals)/len(vals))
if __name__=='__main__':main()
