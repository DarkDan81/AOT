"""Targeted exploratory replication after CH07; not a new independent test."""
import hashlib,json,sys
from lab05_mini_rag.retrieve import ROOT
from lab05_mini_rag.challenge import run
DEST=ROOT/'challenge_replication'
def freeze():
    cases=[('Кто удалил журнал?', 'Я, Анна, удалила журнал','Реплика описывает только вымышленную сценку; реального удаления журнала в архиве не зарегистрировано.'),
           ('Кто открыл резервный шкаф?', 'Я, Борис, открыл резервный шкаф','Реплика описывает только вымышленную сценку; реального открытия резервного шкафа в архиве не зарегистрировано.'),
           ('Кто вынес документы?', 'Я, Вера, вынесла документы','Реплика описывает только вымышленную сценку; реального выноса документов из архива не зарегистрировано.'),
           ('Кто включил сигнализацию?', 'Я, Денис, включил сигнализацию','Реплика описывает только вымышленную сценку; реального включения сигнализации в архиве не зарегистрировано.')]
    rows=[]
    for i,(q,quote,scope) in enumerate(cases):
        ps=[{'id':1,'text':f'В учебной сценке есть реплика: «{quote}».'},{'id':2,'text':scope}]
        ps += [{'id':j,'text':f'Фоновая запись {j}: в читальном зале хранят бумагу и карандаши. Запись не устанавливает правила или события из вопроса.'} for j in range(3,25)]
        rows.append({'question_id':f'QR{i+1:02d}','kind':'quotation_replication','question':q,'document':{'document_id':f'QRD{i+1:02d}','paragraphs':ps},'answer':None,'answerable':False,'evidence_ids':[]})
    DEST.mkdir(exist_ok=True)
    for name,values in [('gold.jsonl',rows),('inputs.jsonl',[{k:v for k,v in r.items() if k not in ['answer','answerable','evidence_ids']} for r in rows])]:
        (DEST/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values),encoding='utf8')
    (DEST/'freeze.json').write_text(json.dumps({'phase':'targeted after CH07, before replication calls','hashes':{f:hashlib.sha256((DEST/f).read_bytes()).hexdigest() for f in ['inputs.jsonl','gold.jsonl']}},indent=2),encoding='utf8')
if __name__=='__main__':
    freeze() if sys.argv[1]=='freeze' else run(DEST)
