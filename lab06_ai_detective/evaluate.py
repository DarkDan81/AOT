import json,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).parent))
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
summary=df.groupby(['kind','mode'])[cols].mean().reset_index();summary=summary.merge(df.groupby(['kind','mode']).agg(n_requests=('run_id','size'),n_valid=('valid','sum')).reset_index(),on=['kind','mode']);summary.to_csv(P/'results/summary.csv',index=False)
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
recognition=[]
for (kind,mode),g in df.groupby(['kind','mode']):
 for status in ('both','insufficient'):
  tp=int(((g.gold_status==status)&(g.pred_status==status)).sum())
  predicted=int((g.pred_status==status).sum());actual=int((g.gold_status==status).sum())
  recognition.append(dict(kind=kind,mode=mode,status=status,true_positive=tp,predicted=predicted,actual=actual,precision=tp/predicted if predicted else None,recall=tp/actual if actual else None))
pd.DataFrame(recognition).to_csv(P/'results/status_recognition.csv',index=False)
# Paired B/C comparison, same retrieval candidates in every pair.
pairs=[]
lookup={(r['variant_id'],r['mode']):r for r in rows}
for r in rows:
 if r['mode']!='B':continue
 c=lookup.get((r['variant_id'],'C'))
 if not c:continue
 assert set(r['context_claim_ids'])==set(c['context_claim_ids'])
 pairs.append(dict(variant_id=r['variant_id'],kind=r['kind'],same_claim_ids=True,status_disagreement=r.get('answer',{}).get('status')!=c.get('answer',{}).get('status'),c_minus_b_accuracy=c.get('metrics',{}).get('status_correct',0)-r.get('metrics',{}).get('status_correct',0),c_minus_b_evidence_f1=c.get('metrics',{}).get('evidence_f1',0)-r.get('metrics',{}).get('evidence_f1',0)))
pd.DataFrame(pairs).to_csv(P/'results/bc_comparison.csv',index=False)
print(summary.to_string(index=False))


