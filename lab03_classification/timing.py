"""Optional 20 serial train-only timing calls, outside test/stress quality metrics."""
import json
from datetime import datetime,timezone
from time import perf_counter
import pandas as pd
from common.llm import complete
from .run import HERE,Sentiment

def main():
    rows=pd.read_csv(HERE/'data/train.csv').head(10)
    path=HERE/'results/serial_timing.jsonl'
    if path.exists():raise FileExistsError('Preserve prior timing log; explicit new experiment required')
    started=perf_counter()
    for row in rows.to_dict('records'):
        for mode in ['zero_shot','few_shot']:
            prompt=(HERE/f'prompts/{mode}.txt').read_text(encoding='utf-8')
            response=complete([{'role':'system','content':prompt},{'role':'user','content':row['text']}],schema=Sentiment,max_tokens=64)
            with path.open('a',encoding='utf-8') as f:f.write(json.dumps({'id':row['id'],'set':'train_timing_only','mode':mode,'workers':1,**response},ensure_ascii=False)+'\n')
    elapsed=perf_counter()-started
    (HERE/'results/serial_timing_summary.json').write_text(json.dumps({'n':20,'wall_seconds':elapsed,'responses_per_second':20/elapsed,'mean_end_to_end_seconds':elapsed/20,'note':'first10train texts, two prompts; separate from quality metrics'},indent=2),encoding='utf-8')

if __name__=='__main__':main()
