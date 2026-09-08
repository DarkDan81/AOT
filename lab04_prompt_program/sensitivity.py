"""Post-hoc gold audit: three text-based corrections, unchanged persisted predictions."""
import csv,hashlib,json
from .extract import ROOT,FIELDS
from .evaluate import score

CORRECTIONS={
    '1037212.tsv': {'field':'indication','remove':['спокойной'],'reason':'Стала спокойной описывает достигнутый результат после приёма, а не исходное показание. Positive_effect сохраняется.'},
    '2500038.tsv': {'field':'indication','remove':['ОРВИ','обструктивным бронхитом','бронхиальную астму'],'reason':'Перечисление введено как итог после курса; текст не называет эти заболевания показаниями до приёма. В adverse_reaction автоматически не переносится: причинная связь и категория требуют отдельного решения.'},
    '895731.tsv': {'field':'drug','remove':['фуфломицин'],'reason':'Это оценочное обозначение, а не конкретное имя препарата. Тилорон и Амиксин сохраняются.'},
}

def main():
    path=ROOT/'data/reviews.jsonl';original=path.read_bytes();original_hash=hashlib.sha256(original).hexdigest()
    assert original_hash==(ROOT/'data/gold.sha256').read_text().strip()
    reviews=[json.loads(s) for s in original.decode('utf8').splitlines()]
    audit=[]
    for r in reviews:
        if r['id'] not in CORRECTIONS:continue
        c=CORRECTIONS[r['id']];before=list(r['gold'][c['field']])
        assert set(c['remove']).issubset(before)
        r['gold'][c['field']]=[v for v in before if v not in c['remove']]
        # Remove only evidence no longer used by any remaining field; no new labels.
        retained={v for f in FIELDS for v in r['gold'][f]}
        r['gold']['evidence']=[v for v in r['gold']['evidence'] if v in retained]
        audit.append({'id':r['id'],'text':r['text'],**c,'before':before,'after':r['gold'][c['field']]})
    reviewed=ROOT/'data/gold_reviewed.jsonl'
    reviewed.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in reviews),encoding='utf8')
    g={r['id']:r['gold'] for r in reviews}
    predictions=list(map(json.loads,(ROOT/'results/predictions.jsonl').read_text(encoding='utf8').splitlines()))
    assert len(predictions)==100
    original_metrics={r['version']:r for r in json.loads((ROOT/'results/metrics.json').read_text(encoding='utf8'))}
    metrics=[]
    for version in ('version_1','version_2'):
        totals=[0,0,0]
        for r in predictions:
            if r['version']!=version:continue
            counts=score(r['parsed'] or {f:[] for f in FIELDS},g[r['id']])
            totals=[a+b for a,b in zip(totals,counts)]
        tp,fp,fn=totals;precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn);f1=2*precision*recall/max(1e-12,precision+recall)
        metrics.append({'version':version,'n':50,'original_f1':original_metrics[version]['f1'],'reviewed_precision':precision,'reviewed_recall':recall,'reviewed_f1':f1,'delta_f1':f1-original_metrics[version]['f1'],'tp':tp,'fp':fp,'fn':fn})
    (ROOT/'results/sensitivity.json').write_text(json.dumps({'phase':'post-hoc independent text audit, no new model calls','original_gold_sha256':original_hash,'reviewed_gold_sha256':hashlib.sha256(reviewed.read_bytes()).hexdigest(),'corrections':audit,'metrics':metrics},ensure_ascii=False,indent=2),encoding='utf8')
    with (ROOT/'results/sensitivity.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    assert path.read_bytes()==original,'Frozen original must remain unchanged'
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
