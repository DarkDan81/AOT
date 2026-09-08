from pathlib import Path
import pandas as pd
p=Path('lab02_annotation')
d=pd.read_csv(p/'data/disputed30.csv')
positive={'T10','T15','T17','T19','T26','T31','T32','T38','T41','T45','T49','T56','T66','T70'}
uncertain={'T65'}
d['label']=['1' if i in positive else '?' if i in uncertain else '0' for i in d.id]
d['reason']=d.label.map({'1':'Действующее требование ускорения, короткий срок либо авария по общей инструкции.','0':'Нет отдельного сигнала ускорения: план, ожидание или пересказ.','?':'Остановка работы без ясного актуального требования и последствий.'})
d[['id','label','reason']].to_csv(p/'annotations/revised_a.csv',index=False)
