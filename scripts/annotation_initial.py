from pathlib import Path
import pandas as pd
p=Path('lab02_annotation')
labels='''1 0 1 0 1 ? ? 1 0 ?
1 0 1 0 0 ? 1 ? 0 0
1 ? 0 1 0 ? ? 0 1 0
? 0 ? 1 ? 0 1 ? 0 0
? 0 1 ? ? ? 1 0 1 ?
0 1 1 ? 0 ? ? 0 1 ?
0 0 1 0 ? ? 0 0 1 ?
? 1 ? 0 ? ? 0 1 0 0
0 1 0 ? 1 0 1 1 0 0'''.split()
assert len(labels)==90
reasons={'1':'Явная опасность либо близкий срок с последствиями задержки.','0':'Нет действующего безотлагательного запроса по определению A.','?':'Недостаточно контекста о сроке или последствиях; одного усилителя недостаточно.'}
pd.DataFrame({'id':[f'T{i:02d}' for i in range(1,91)],'label':labels,'reason':[reasons[v] for v in labels]}).to_csv(p/'annotations/initial_a.csv',index=False)
d=pd.read_csv(p/'data/texts.csv')
for a in 'abc':
    x=pd.read_csv(p/f'annotations/initial_{a}.csv',dtype=str)
    d=d.merge(x[['id','label']].rename(columns={'label':f'annotator_{a}'}),on='id',validate='one_to_one')
cols=[f'annotator_{a}' for a in 'abc']
d['disagreement']=d[cols].apply(lambda r:sum(r.iloc[i]!=r.iloc[j] for i in range(3) for j in range(i+1,3)),axis=1)
(p/'results').mkdir(exist_ok=True)
d.to_csv(p/'results/initial_combined.csv',index=False)
selected=d.sort_values(['disagreement','id'],ascending=[False,True]).head(30)
selected[['id','text']].to_csv(p/'data/disputed30.csv',index=False)
print(selected.to_string(index=False))
