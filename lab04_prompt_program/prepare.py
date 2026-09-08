"""Freeze 50 shortest complete RuDReC reviews and an explicitly adapted reference."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent
FIELDS = ['drug', 'indication', 'positive_effect', 'adverse_reaction', 'evidence']

def main():
    groups = collections.defaultdict(list)
    for line in (ROOT/'data/source.json').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)  # upstream contains NaN in unused concept_name
        groups[row['file_name']].append(row)
    chosen = sorted(groups.items(), key=lambda kv: (sum(len(r['text']) for r in kv[1]), kv[0]))[:50]
    positive = {
        0:['быстро сбивает температуру'], 1:['успокаивают'],4:['действительно помогает'],
        7:['Помогла'],8:['легче'],10:['быстрее побороть недуг','повышает аппетит'],
        11:['Начал помогать быстро','Не дает подняться температуре'],20:['Стала спокойной'],
        24:['Помогает','оказывает положительное действие'],26:['жизнь наладилась'],
        32:['что-то из них мне помогло'],34:['Самые эффективные'],36:['стало лучше'],
        37:['помог тут же'],40:['Эффект положительный'],42:['высокую эффективность'],
        43:['прошел озноб и жар, усталость'],44:['помогал','хорошо справляется с бессонницей'],
        45:['стало полегче горлу'],
    }
    rows=[]
    for i,(id,sentences) in enumerate(chosen):
        text=''.join(s['text'] for s in sorted(sentences,key=lambda s:s['sentence_id']))
        gold={k:[] for k in FIELDS}
        for s in sentences:
            for e in s['entities']:
                field={'Drugname':'drug','DI':'indication','ADR':'adverse_reaction'}.get(e['entity_type'])
                if field and e['entity_text'] in text and e['entity_text'] not in gold[field]:
                    gold[field].append(e['entity_text'])
        gold['positive_effect']=positive.get(i,[])
        # Review-level task differs from corpus NER: remove hypothetical/quoted ADR;
        # retain directly observed effects in author or explicitly described patient.
        if i==10: gold['adverse_reaction']=[]
        if i==15: gold['adverse_reaction']=['слабость','сонливость','общее недомогание']
        if i==18: gold['adverse_reaction']=['высыпали на лице пятна']
        if i==27: gold['adverse_reaction']=['сон как отбило','сухость во рту','чувство тошноты']
        if i==39: gold['adverse_reaction']=[]
        if i==16: gold['drug']=['виферона']
        if i==36: gold['drug']=['супрастина']
        if i==44: gold['adverse_reaction']=['сильно много побочных','имеет побочных меньше чем аминазин']
        gold['evidence']=list(dict.fromkeys(v for k in FIELDS[:-1] for v in gold[k]))
        assert all(v in text for v in gold['evidence'])
        rows.append({'id':id,'text':text,'gold':gold})
    payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    (ROOT/'data/reviews.jsonl').write_bytes(payload.encode('utf-8'))
    (ROOT/'data/gold.sha256').write_text(hashlib.sha256(payload.encode()).hexdigest()+'\n',encoding='utf-8')

if __name__=='__main__':main()
