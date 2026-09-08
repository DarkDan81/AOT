from pathlib import Path
import pandas as pd,json
p=Path('lab03_classification');df=pd.read_csv(p/'data/train.csv')
# Deterministic train-only selection. No test example is consulted.
selected=df[df.text.str.len().between(25,100)].groupby('label',sort=True).head(2)
assert selected.label.value_counts().to_dict()=={'negative':2,'neutral':2,'positive':2}
prompt=(p/'prompts/zero_shot.txt').read_text(encoding='utf-8')+'\n\nПримеры:\n'
for row in selected.itertuples():prompt+='Текст: '+row.text+'\nОтвет: '+json.dumps({'label':row.label},ensure_ascii=False)+'\n\n'
(p/'prompts/few_shot.txt').write_text(prompt,encoding='utf-8')
selected.to_csv(p/'data/fewshot_examples.csv',index=False)
