"""Fixed three-class subset of the original RuSentiment public mirror."""
from pathlib import Path
import hashlib, json
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://raw.githubusercontent.com/strawberrypie/rusentiment/master/Dataset/'

def main():
    raw = ROOT/'tmp/rusentiment'; raw.mkdir(parents=True, exist_ok=True)
    frames = {}
    checks = {}
    for key, name in [('random','rusentiment_random_posts.csv'),('test','rusentiment_test.csv')]:
        path = raw/f'{key}.csv'
        if not path.exists():
            response=requests.get(URL+name,timeout=120); response.raise_for_status(); path.write_bytes(response.content)
        df=pd.read_csv(path); checks['source_'+key]={'rows':len(df),'nulls':df.isna().sum().to_dict(),'duplicate_texts':int(df.text.duplicated().sum()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'url':URL+name}
        df['id']=[f'{key}_{i:05d}' for i in range(len(df))]
        frames[key]=df.dropna(subset=['text','label']).drop_duplicates('text').query("label in ['positive','negative','neutral']")[['id','text','label']]
    train_pool=frames['random'][~frames['random'].text.isin(frames['test'].text)]
    selected,_=train_test_split(train_pool,train_size=3500,stratify=train_pool.label,random_state=42)
    train,val=train_test_split(selected,test_size=500,stratify=selected.label,random_state=42)
    test,_=train_test_split(frames['test'],train_size=500,stratify=frames['test'].label,random_state=42)
    target=ROOT/'lab03_classification/data'; target.mkdir(parents=True,exist_ok=True)
    for name,df in [('train',train),('validation',val),('test',test)]:
        df.to_csv(target/f'{name}.csv',index=False)
        checks[name]={'rows':len(df),'classes':df.label.value_counts().to_dict(),'sha256':hashlib.sha256((target/f'{name}.csv').read_bytes()).hexdigest()}
    sample,_=train_test_split(train,train_size=150,stratify=train.label,random_state=42)
    sample.to_csv(ROOT/'lab01_representations/data/texts.csv',index=False)
    for a,b in [(train,val),(train,test),(val,test)]: assert not set(a.text)&set(b.text)
    checks['policy']='Three original classes; speech and skip excluded. Unique text. Original held-out test preserved. Stratified seed42, no test tuning.'
    (target/'provenance.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(checks,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

