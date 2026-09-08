import csv
import json
import re
from collections import Counter
from .extract import ROOT,FIELDS

def tokens(values):return set(re.findall(r'\w+',' '.join(values).lower()))

def score(pred,gold):
    p={(f,t) for f in FIELDS for t in tokens(pred[f])}
    g={(f,t) for f in FIELDS for t in tokens(gold[f])}
    return len(p&g),len(p-g),len(g-p)

def main():
    gold={r['id']:r for r in map(json.loads,(ROOT/'data/reviews.jsonl').read_text(encoding='utf-8').splitlines())}
    rows=list(map(json.loads,(ROOT/'results/predictions.jsonl').read_text(encoding='utf-8').splitlines()))
    metrics=[]; errors=[]
    for version in ('version_1','version_2'):
        subset=[r for r in rows if r['version']==version]
        c=Counter()
        for r in subset:
            first=r['attempts'][0]
            try:
                initial=json.loads(first['content']); c['json_first_valid']+=1
            except (ValueError,TypeError):initial={}
            c['first_contract_valid']+=first['error'] is None
            c['final_valid']+=r['valid']; c['retries']+=len(r['attempts'])-1
            for a in r['attempts']:
                c['seconds']+=a.get('elapsed_seconds',0)
                c['prompt_tokens']+=a.get('usage',{}).get('prompt_tokens',0)
                c['completion_tokens']+=a.get('usage',{}).get('completion_tokens',0)
            pred=r['parsed'] or {f:[] for f in FIELDS}
            tp,fp,fn=score(pred,gold[r['id']]['gold'])
            c['tp']+=tp;c['fp']+=fp;c['fn']+=fn
            c['unsupported_final']+=sum(v not in gold[r['id']]['text'] for f in FIELDS for v in pred[f])
            c['unsupported_initial']+=sum(not isinstance(v,str) or v not in gold[r['id']]['text'] for f in FIELDS for v in initial.get(f,[]) if isinstance(initial.get(f,[]),list)) if isinstance(initial,dict) else 0
            if fp or fn:errors.append({'id':r['id'],'version':version,'tp':tp,'fp':fp,'fn':fn,'text':gold[r['id']]['text'],'gold':gold[r['id']]['gold'],'pred':pred})
        precision=c['tp']/max(1,c['tp']+c['fp']);recall=c['tp']/max(1,c['tp']+c['fn'])
        metrics.append({'version':version,'n':len(subset),**dict(c),'precision':precision,'recall':recall,'f1':2*precision*recall/max(1e-12,precision+recall)})
    (ROOT/'results/metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'results/errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2),encoding='utf-8')
    with (ROOT/'results/metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(m.keys() for m in metrics))));writer.writeheader();writer.writerows(metrics)
    print(json.dumps(metrics,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
