"""Run 4 sentiment systems on frozen test and synthetic stress sets."""
import argparse, hashlib, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime,timezone
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from common.llm import complete, parse_json

HERE=Path(__file__).resolve().parent
class Sentiment(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    label:Literal['positive','negative','neutral']

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--classical-only',action='store_true'); parser.add_argument('--workers',type=int,choices=[1,2],default=2); args=parser.parse_args()
    out=HERE/'results';out.mkdir(exist_ok=True)
    train=pd.read_csv(HERE/'data/train.csv'); val=pd.read_csv(HERE/'data/validation.csv')
    assert hashlib.sha256((HERE/'data/stress.csv').read_bytes()).hexdigest()==(HERE/'data/stress.sha256').read_text().strip(),'Stress gold changed after freeze'
    provenance=json.loads((HERE/'data/provenance.json').read_text(encoding='utf-8'))
    for split in ['train','validation','test']:
        assert hashlib.sha256((HERE/f'data/{split}.csv').read_bytes()).hexdigest()==provenance[split]['sha256'],f'{split} changed after freeze'
    tuning=[]; models=[]
    # Exactly one tuned hyperparameter, C; all vectorizer parameters fixed.
    for c in [0.5,1.0,2.0]:
        model=Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2))),('lr',LogisticRegression(C=c,max_iter=1000,random_state=42))])
        model.fit(train.text,train.label); pred=model.predict(val.text)
        tuning.append({'C':c,'validation_macro_f1':f1_score(val.label,pred,average='macro')}); models.append(model)
    best=int(np.argmax([x['validation_macro_f1'] for x in tuning])); model=models[best]
    pd.DataFrame(tuning).to_csv(out/'validation_tuning.csv',index=False)
    (out/'selected_parameters.json').write_text(json.dumps({'C':tuning[best]['C'],'ngram_range':[1,2],'random_state':42,'selection':'validation macro F1; first wins ties'},indent=2))
    majority=train.label.value_counts().index[0]
    frames=[]
    for name in ['test','stress']:
        path=HERE/f'data/{name}.csv'
        if not path.exists():
            if args.classical_only:continue
            raise FileNotFoundError(path)
        df=pd.read_csv(path);df['set']=name
        t=time.perf_counter();df['majority']=majority;df['majority_seconds']=(time.perf_counter()-t)/len(df)
        # Single-text latency is measured explicitly (batch throughput is a different metric).
        predictions=[];latencies=[]
        for text in df.text:
            t=time.perf_counter();predictions.append(model.predict([text])[0]);latencies.append(time.perf_counter()-t)
        df['tfidf_lr']=predictions;df['tfidf_lr_seconds']=latencies;frames.append(df)
    data=pd.concat(frames,ignore_index=True)
    data.to_csv(out/'classical_predictions.csv',index=False)
    if args.classical_only:return
    prompts={m:(HERE/f'prompts/{m}.txt').read_text(encoding='utf-8') for m in ['zero_shot','few_shot']}
    rawpath=out/'llm_responses.jsonl'
    existing={}
    if rawpath.exists():
        for line in rawpath.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);existing[(r['id'],r['mode'])]=r
    # Two concurrent requests are optional. HTTP latency includes server scheduling;
    # it must not be described as exclusive sequential latency.
    def classify(row,mode):
        response=complete([{'role':'system','content':prompts[mode]},{'role':'user','content':row['text']}],schema=Sentiment,max_tokens=64)
        r={'id':row['id'],'set':row['set'],'mode':mode,'client_workers':args.workers,**response}
        try:r['label']=Sentiment.model_validate(parse_json(response['content'])).label;r['valid']=True
        except (ValueError,TypeError) as e:r['label']='invalid';r['valid']=False;r['error']=str(e)
        return r
    inference_start=time.perf_counter(); started_at=datetime.now(timezone.utc).isoformat(); new_count=0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending=[pool.submit(classify,row,mode) for row in data.to_dict('records') for mode in prompts if (row['id'],mode) not in existing]
        for future in as_completed(pending):
            r=future.result()
            with rawpath.open('a',encoding='utf-8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
            existing[(r['id'],r['mode'])]=r
            new_count+=1
            if len(existing)%40==0:print(f'{len(existing)}/1200 LLM responses',flush=True)
    inference_seconds=time.perf_counter()-inference_start
    execution={'started_at':started_at,'workers':args.workers,'new_responses':new_count,'wall_seconds':inference_seconds,'responses_per_second':new_count/inference_seconds if new_count else None,'cached_responses':len(existing)-new_count}
    with (out/'execution_runs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(execution)+'\n')
    assert len(existing)==len(data)*len(prompts),'Incomplete or foreign response cache; do not report partial metrics'
    for mode in prompts:
        data[mode]=[existing[(i,mode)]['label'] for i in data.id]
        for column in ['elapsed_seconds','prompt_tokens','completion_tokens']:
            data[f'{mode}_{column}']=[(existing[(i,mode)].get('usage',{}).get(column,0) if 'tokens' in column else existing[(i,mode)][column]) for i in data.id]
    data.to_csv(out/'predictions.csv',index=False)
    metrics=[]
    for name,df in data.groupby('set'):
        for mode in ['majority','tfidf_lr','zero_shot','few_shot']:
            llm=mode in prompts
            metrics.append({'set':name,'method':mode,'n':len(df),'accuracy':accuracy_score(df.label,df[mode]),'macro_f1':f1_score(df.label,df[mode],labels=['positive','negative','neutral'],average='macro',zero_division=0),'mean_seconds':df[f'{mode}_elapsed_seconds' if llm else f'{mode}_seconds'].mean(),'input_tokens':int(df[f'{mode}_prompt_tokens'].sum()) if llm else 0,'output_tokens':int(df[f'{mode}_completion_tokens'].sum()) if llm else 0,'api_cost_per_1000_usd':0,'cost_note':'local inference, no API tariff; electricity/hardware not estimated'})
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    print(pd.DataFrame(metrics).to_string(index=False))
if __name__=='__main__':main()
