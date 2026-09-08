"""Recompute quality and choose transparent error candidates from persisted responses."""
import json
import sys
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
from .run import HERE

def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    data=pd.read_csv(HERE/'results/predictions.csv')
    assert len(data)==600 and data.id.nunique()==600
    assert len(data[data['set']=='test'])==500 and len(data[data['set']=='stress'])==100
    metrics=[]
    for category,df in data[data['set']=='stress'].groupby('category'):
        for mode in ['majority','tfidf_lr','zero_shot','few_shot']:
            metrics.append({'category':category,'method':mode,'n':len(df),'accuracy':accuracy_score(df.label,df[mode]),'macro_f1_three_labels':f1_score(df.label,df[mode],labels=['positive','negative','neutral'],average='macro',zero_division=0)})
    pd.DataFrame(metrics).to_csv(HERE/'results/stress_category_metrics.csv',index=False)
    chosen=[];used=set()
    for group,mask in [('classical_error',data.label!=data.tfidf_lr),('llm_error',data.label!=data.few_shot),('disagreement',data.tfidf_lr!=data.few_shot)]:
        candidates=data[mask & ~data.id.isin(used)].head(5)
        assert len(candidates)==5,f'Only{len(candidates)}real cases for{group}; do not invent failures'
        for row in candidates.to_dict('records'):
            row['analysis_group']=group;chosen.append(row);used.add(row['id'])
    pd.DataFrame(chosen).to_csv(HERE/'results/error_cases_15.csv',index=False)
    print(pd.DataFrame(chosen)[['analysis_group','id','text','label','tfidf_lr','few_shot']].to_string(index=False))

if __name__=='__main__':main()
