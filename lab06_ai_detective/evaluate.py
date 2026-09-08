import json,pathlib
import pandas as pd
from src.evaluation import evidence_metrics
P=pathlib.Path(__file__).parent

def read(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]
rows=read(P/'results/responses.jsonl')
flat=[]
for r in rows:
 a=r.get('answer',{});t=r.get('transport',{});u=t.get('usage',{})
 flat.append(dict(run_id=r['run_id'],kind=r['kind'],mode=r['mode'],question_id=r['question_id'],gold_status=r['gold_status'],pred_status=a.get('status','INVALID'),valid=r['valid'],confidence=a.get('confidence'),seconds=t.get('elapsed_seconds',r.get('elapsed_seconds')),input_tokens=u.get('prompt_tokens',0),output_tokens=u.get('completion_tokens',0),**r.get('metrics',{'status_correct':0})))
df=pd.DataFrame(flat);df.to_csv(P/'results/metrics_per_response.csv',index=False,encoding='utf8')
cols=['status_correct','evidence_precision','evidence_recall','evidence_f1','counterevidence_recall','both_correct','insufficient_correct','confident_when_insufficient','seconds','input_tokens','output_tokens']
summary=df.groupby(['kind','mode'])[cols].mean().reset_index();summary.to_csv(P/'results/summary.csv',index=False)
base={(r['question_id'],r['mode']):r for r in rows if r['kind']=='baseline'}
st=[]
for r in rows:
 if r['kind']=='baseline' or not r['valid']:continue
 b=base.get((r['question_id'],r['mode']))
 if not b or not b['valid']:continue
 a,ba=r['answer'],b['answer']
 st.append(dict(kind=r['kind'],mode=r['mode'],question_id=r['question_id'],status_changed=a['status']!=ba['status'],answer_changed=a['answer']!=ba['answer'],evidence_stability_f1=evidence_metrics(a['evidence_ids'],ba['evidence_ids'])[2],expected_status_correct=r['metrics']['status_correct']))
pd.DataFrame(st).to_csv(P/'results/stability.csv',index=False)
# Calibration by fixed confidence bins; descriptive only for a small synthetic sample.
df['confidence_bin']=pd.cut(df.confidence,[0,.5,.8,1],include_lowest=True)
df.groupby(['mode','confidence_bin'],observed=True).agg(n=('run_id','size'),mean_confidence=('confidence','mean'),accuracy=('status_correct','mean')).to_csv(P/'results/calibration.csv')
print(summary.to_string(index=False))
