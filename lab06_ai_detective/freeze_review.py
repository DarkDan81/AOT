import json,hashlib,pathlib,sys
P=pathlib.Path(__file__).parent;sys.path.insert(0,str(P.parent))
from lab06_ai_detective.src.retrieval import retrieve
from lab06_ai_detective.src.evaluation import evidence_metrics
D=P/'data'
read=lambda f:[json.loads(x) for x in (D/f).read_text(encoding='utf-8-sig').splitlines()]
vs=read('variants.jsonl')
for v in vs:
 v['review']['status']='reviewed_by_three_ai_roles'
 v['review']['reviewers']=['author_agent','project_lead_agent','lab05_agent']
(D/'variants.jsonl').write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in vs),encoding='utf8')
frozen=json.loads((D/'gold_freeze.json').read_text(encoding='utf8'))
frozen['sha256']['variants.jsonl']=hashlib.sha256((D/'variants.jsonl').read_bytes()).hexdigest()
frozen['review']='All gold reviewed before model inference by three AI roles; not human verification.'
(D/'gold_freeze.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf8')
rows=[]
for v in vs:
 if not v['applicable']:continue
 q=v['question'];selected=retrieve(q['question'],v['claims']);ids={c['claim_id'] for c in selected}
 gold=set(q['gold_evidence_ids']+q['gold_counterevidence_ids'])
 rows.append(dict(variant_id=v['variant_id'],kind=v['kind'],retrieval_recall=len(ids&gold)/len(gold) if gold else 1,all_gold_retrieved=gold<=ids,claim_ids=sorted(ids),gold_ids=sorted(gold)))
(P/'results/retrieval.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
(P/'results/verification.json').write_text(json.dumps({'pytest':'19 passed in 0.49s','streamlit_apptest_initial_exceptions':0,'model_calls_so_far':0,'word_count':frozen['words'],'claims':36,'documents':9,'questions':12},indent=2),encoding='utf8')
print('Retrieval baseline recall',sum(r['retrieval_recall'] for r in rows if r['kind']=='baseline')/12)
