"""Run from repository root; resume completed requests without hidden retries."""
import argparse,json,pathlib,sys,time,hashlib
P=pathlib.Path(__file__).parent;sys.path.insert(0,str(P));sys.path.insert(0,str(P.parent))
from lab06_ai_detective.src.context_builder import build
from lab06_ai_detective.src.retrieval import retrieve
from lab06_ai_detective.src.llm import ask
from lab06_ai_detective.src.evaluation import score,evidence_metrics

def read(path):return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines()]

def error_record(error,elapsed):
 row=dict(valid=False,error=type(error).__name__+': '+str(error),elapsed_seconds=elapsed)
 response=getattr(error,'response',None)
 if isinstance(response,dict):row['transport']=response
 elif response is not None:
  row['http_status']=getattr(response,'status_code',None)
  row['http_body']=getattr(response,'text',None)
 return row

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);parser.add_argument('--kind',choices=['baseline','permutation','deletion','contradiction']);args=parser.parse_args()
 out=P/'results';out.mkdir(exist_ok=True);(out/'contexts').mkdir(exist_ok=True)
 target=out/'responses.jsonl';done={r['run_id'] for r in read(target)} if target.exists() else set()
 variants=read(P/'data/variants.jsonl');n=0
 for v in variants:
  if not v['applicable'] or (args.kind and v['kind']!=args.kind):continue
  assert v['review']['status'].startswith('reviewed'), 'Unreviewed variant'
  q=v['question'];selected=retrieve(q['question'],v['claims'])
  for mode in ['A','B','C']:
   rid=v['variant_id']+'_'+mode
   if rid in done:continue
   context,ids=build(mode,q['question'],v['documents'],v['claims'],selected)
   context_path=out/'contexts'/f'{rid}.txt';context_path.write_text(context,encoding='utf8')
   row=dict(run_id=rid,variant_id=v['variant_id'],kind=v['kind'],mode=mode,question_id=q['question_id'],gold_status=q['gold_status'],context_claim_ids=ids,context_document_order=[d['document_id'] for d in v['documents']],context_sha256=hashlib.sha256(context.encode()).hexdigest(),context_file=str(context_path.relative_to(P)),review=v['review'])
   start=time.perf_counter()
   try:
    answer,response=ask(q,context,ids)
    row.update(answer=answer,transport=response,valid=True,metrics=score(answer,q))
   except Exception as e:
    row.update(error_record(e,time.perf_counter()-start))
   with target.open('a',encoding='utf8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
   n+=1;print(rid,'valid='+str(row['valid']),flush=True)
   if args.limit and n>=args.limit:return
if __name__=='__main__':main()
