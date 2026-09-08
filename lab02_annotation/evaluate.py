from pathlib import Path
import itertools,json
import numpy as np,pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa
HERE=Path(__file__).resolve().parent

def metrics(df,stage):
    cols=['annotator_'+a for a in 'abc'];rows=[]
    for a,b in itertools.combinations(cols,2):
        rows.append({'stage':stage,'metric':a+' / '+b,'value':float(cohen_kappa_score(df[a],df[b],labels=['0','1','?'])),'n':len(df)})
    counts=np.array([[list(r).count(l) for l in ['0','1','?']] for r in df[cols].values])
    rows.append({'stage':stage,'metric':'Fleiss kappa','value':float(fleiss_kappa(counts)),'n':len(df)})
    rows.append({'stage':stage,'metric':'full agreement','value':float((df[cols].nunique(axis=1)==1).mean()),'n':len(df)})
    return rows

def main():
    data=pd.read_csv(HERE/'data/texts.csv');primary=data.copy()
    for a in 'abc':
        x=pd.read_csv(HERE/f'annotations/initial_{a}.csv',dtype=str)
        assert len(x)==90 and x.id.is_unique and set(x.label)<={'0','1','?'}
        primary=primary.merge(x[['id','label']].rename(columns={'label':'annotator_'+a}),on='id',validate='one_to_one')
    cols=['annotator_'+a for a in 'abc']
    primary['disagreement']=primary[cols].apply(lambda r:sum(r.iloc[i]!=r.iloc[j] for i in range(3) for j in range(i+1,3)),axis=1)
    selected=primary.sort_values(['disagreement','id'],ascending=[False,True]).head(30)
    causes={
       'explicit_vs_justified':['T10','T26','T31','T38','T41','T45','T56','T66','T70'],
       'advertising':['T15','T19','T32'],
       'deadline_context':['T06','T16','T22','T27','T35','T46','T49','T57'],
       'implicit_request':['T07','T17','T18','T33','T44','T50','T54','T60','T65'],
       'quotation':['T61']}
    inv={i:k for k,v in causes.items() for i in v};selected=selected.copy();selected['cause']=selected.id.map(inv)
    assert selected.cause.notna().all()
    selected.to_csv(HERE/'results/disagreements.csv',index=False)
    revised=selected[['id','text']].copy()
    for a in 'abc':
        x=pd.read_csv(HERE/f'annotations/revised_{a}.csv',dtype=str)
        assert len(x)==30 and set(x.id)==set(selected.id)
        revised=revised.merge(x[['id','label']].rename(columns={'label':'annotator_'+a}),on='id',validate='one_to_one')
    rows=metrics(primary,'initial_90')+metrics(selected,'initial_disputed_30')+metrics(revised,'revised_same_30')
    pd.DataFrame(rows).to_csv(HERE/'results/agreement_metrics.csv',index=False)
    primary.to_csv(HERE/'results/initial_combined.csv',index=False)
    revised.to_csv(HERE/'results/revised_combined.csv',index=False)
    combined=primary.merge(revised[['id']+cols].rename(columns={c:c+'_revised' for c in cols}),on='id',how='left')
    combined.to_csv(HERE/'results/annotations_all.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
