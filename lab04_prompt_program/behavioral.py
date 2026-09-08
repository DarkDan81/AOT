"""Real-backend behavioral acceptance tests, separate from deterministic unit tests."""
import json
from .extract import ROOT,extract

CASES=[
 ('empty','',{'drug':[],'indication':[],'positive_effect':[],'adverse_reaction':[]}),
 ('no-drug','Средство помогло от кашля.',{'drug':[]}),
 ('multiple-drugs','Пила Аспирин, потом Парацетамол.',{'drug':['Аспирин','Парацетамол']}),
 ('negation','Аспирин от боли не помог.',{'positive_effect':[]}),
 ('hypothesis','Может, Аспирин поможет от боли.',{'positive_effect':[]}),
 ('other-patient','У моего ребенка после Аспирина появилась сыпь.',{'adverse_reaction':['сыпь']}),
 ('quotation','В чужом отзыве написано: «Аспирин вызвал сыпь». Я принимал его без побочных реакций.',{'adverse_reaction':[]}),
 ('typo','Приняла Асперин от боли.',{'drug':['Асперин']}),
]

def main():
    path=ROOT/'results/behavioral.jsonl'
    done=set()
    if path.exists():done={json.loads(x)['case'] for x in path.read_text(encoding='utf-8').splitlines()}
    for name,text,expected in CASES:
        if name in done:continue
        result=extract(text)
        p=result['parsed'] or {}
        passed=result['valid'] and all((p.get(k)==[] if not v else all(any(x in y for y in p.get(k,[])) for x in v)) for k,v in expected.items())
        row={'case':name,'text':text,'expected_constraints':expected,'passed':passed,**result}
        with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(name,passed,flush=True)

if __name__=='__main__':main()
