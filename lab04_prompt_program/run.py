"""Run from repository root: python -m lab04_prompt_program.run."""
import hashlib
import json
from pathlib import Path
from .extract import extract,ROOT

def main():
    data=(ROOT/'data/reviews.jsonl').read_bytes()
    assert hashlib.sha256(data).hexdigest()==(ROOT/'data/gold.sha256').read_text().strip()
    rows=[json.loads(x) for x in data.decode().splitlines()]
    output=ROOT/'results/predictions.jsonl'
    done=set()
    if output.exists():
        done={(r['id'],r['version']) for r in map(json.loads,output.read_text(encoding='utf-8').splitlines())}
    for row in rows:
        for version in ('version_1','version_2'):
            if (row['id'],version) in done:continue
            result=extract(row['text'],version)
            result.update(id=row['id'],version=version)
            with output.open('a',encoding='utf-8') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
            print(row['id'],version,result['valid'],flush=True)

if __name__=='__main__':main()
