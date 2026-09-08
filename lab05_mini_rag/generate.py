import argparse, hashlib, json, sys
from pathlib import Path
from pydantic import BaseModel, ConfigDict, model_validator
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT.parent))
from common.llm import complete, parse_json
from lab05_mini_rag.retrieve import read, top5
class Answer(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    answer:str|None
    evidence_ids:list[int]
    answerable:bool
    @model_validator(mode='after')
    def coherent(self):
        if self.answerable and (not self.answer or not self.evidence_ids):raise ValueError('answerable needs answer and evidence')
        if not self.answerable and (self.answer is not None or self.evidence_ids):raise ValueError('refusal needs null and empty evidence')
        if len(set(self.evidence_ids))!=len(self.evidence_ids):raise ValueError('duplicate evidence')
        return self
def validate(content, ids):
    a=Answer.model_validate(parse_json(content))
    if not set(a.evidence_ids).issubset(ids):raise ValueError('evidence outside supplied context')
    return a.model_dump()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);args=parser.parse_args()
    manifest=json.loads((ROOT/'data/freeze_manifest.json').read_text())
    # Generator deliberately never opens test_gold: only public inputs are verified/read.
    for p,h in manifest.items():
        if p.startswith('evaluation_private/'):continue
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    assert (ROOT/'results/retrieval_before_llm.json').exists(),'compute recall before LLM'
    docs={x['document_id']:x for x in read(ROOT/'data/documents.jsonl')}
    qs=read(ROOT/'data/questions.jsonl'); prompt=(ROOT/'prompt.txt').read_text(encoding='utf-8')
    output=ROOT/'results/predictions.jsonl'; existing=read(output) if output.exists() else []
    done={(x['question_id'],x['mode']) for x in existing};n=0
    for q in qs:
        for mode in ['full','bm25']:
            if (q['question_id'],mode) in done:continue
            ps=docs[q['document_id']]['paragraphs'] if mode=='full' else top5(docs[q['document_id']],q['question'])
            context='\n\n'.join(f"[{p['id']}] {p['text']}" for p in ps)
            r=complete([{'role':'system','content':prompt},{'role':'user','content':f"Вопрос: {q['question']}\nКонтекст:\n{context}"}],schema=Answer.model_json_schema(),max_tokens=700)
            row={**q,'mode':mode,'context_ids':[p['id'] for p in ps],**r}
            try:row.update(prediction=validate(r['content'],set(row['context_ids'])),valid=True,error=None)
            except Exception as e:row.update(prediction=None,valid=False,error=str(e))
            with output.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            print(q['question_id'],mode,row['valid'],flush=True);n+=1
            if args.limit and n>=args.limit:return
if __name__=='__main__':main()
