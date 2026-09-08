"""Create reviewed single-factor variants before inference; never use hidden truth."""
import copy,json,random,pathlib
P=pathlib.Path(__file__).parent
D=P/'data'
def read(name):return [json.loads(x) for x in (D/name).read_text(encoding='utf8').splitlines()]
docs,claims,questions=read('documents.jsonl'),read('claims.jsonl'),read('questions.jsonl')
remove={'Q03':'C06','Q04':'C28','Q05':'C06','Q06':'C04'}
# One sentence from one additional source. Existing material is unchanged.
contradictions={
'Q01':'Сверенный независимый журнал сообщает: вход Лады зарегистрирован в 18:13, а запись 18:12 ошибочна.',
'Q02':'Независимый аудит сообщает: первый выпуск не был опубликован службой «Шлюз».',
'Q03':'Независимый журнал прав сообщает: Лада уже имела право публикации в 18:15.',
'Q04':'Независимый журнал очереди сообщает: первое задание поставлено в очередь в 18:05, а не в 17:55.',
'Q05':'Независимый журнал прав сообщает: право публикации Ладе выдано в 18:05, а не в 18:20.',
'Q06':'Независимый журнал публикации сообщает: контрольная сборка опубликована в 18:23, а не в 18:25.',
'Q07':'Координатор второго дежурства сообщает: резервный ключ в 18:25 был выключен.',
'Q08':'Вторая заверенная редакция регламента сообщает: для первого выпуска резервное исключение не требовало ключа и разрешения.',
'Q09':'Сава сообщил: «В 18:15 Лада находилась в серверной».',
'Q10':'Сава сообщил: «В 18:15 индикатор был красным».',
'Q11':'Вера в поздней записи сообщила: «Намерения Ильи мне известны, но я не указываю их в этом сообщении».',
'Q12':'Вера в поздней записи сообщила: «Личность скачавшего архив установлена, но в этой записи я её не называю».'}
all_variants=[]
for q in questions:
 for kind in ['baseline','permutation','deletion','contradiction']:
  vd,vc,vq=copy.deepcopy(docs),copy.deepcopy(claims),copy.deepcopy(q)
  applicable=True;changed=[];reason=''
  if kind=='permutation':random.Random(42).shuffle(vd);reason='Только порядок документов; содержание и эталон сохраняются.'
  if kind=='deletion':
   if q['question_id'] not in remove:
    applicable=False;reason='Удаление необходимой посылки задано только для multi-hop и временных вопросов; не рассчитывается.'
   else:
    cid=remove[q['question_id']];c=next(x for x in vc if x['claim_id']==cid)
    for d in vd:
     if d['document_id']==c['document_id']:
      assert d['text'].count(c['evidence'])==1
      d['text']=d['text'].replace(c['evidence'],'')
    vc=[x for x in vc if x['claim_id']!=cid];changed=[cid]
    vq.update(gold_status='insufficient',gold_answer=None,gold_evidence_ids=[x for x in q['gold_evidence_ids'] if x!=cid],gold_counterevidence_ids=[])
    reason='Удалена одна необходимая посылка и её дословный фрагмент из документа. Оставшиеся посылки не определяют сравниваемое время.'
  if kind=='contradiction':
   sentence=contradictions[q['question_id']]
   vd.append(dict(document_id='D10',title='Конфликтующая независимая запись',source_type='statement',created_at='2026-03-14T20:00:00',author='Дополнительный источник',text=sentence+' Источники имеют одинаковую заявленную надёжность; независимого способа предпочесть один из них в досье нет.'))
   vc.append(dict(claim_id='C37',text=sentence,document_id='D10',evidence=sentence,status='disputed',path=[1,2,1],event_time='99:99'));changed=['C37']
   if q['gold_status'] in ('entailed','contradicted'):
    vq.update(gold_status='both',gold_answer='Исходное основание и добавленный источник несовместимы; приоритет не установлен.',gold_counterevidence_ids=['C37'])
   if q['gold_status']=='contradicted':
    vq['gold_evidence_ids']=['C37'];vq['gold_counterevidence_ids']=q['gold_evidence_ids']
   elif q['gold_status']=='both':vq['gold_evidence_ids']=q['gold_evidence_ids']+['C37']
   if q['question_id']=='Q03':vq['gold_evidence_ids']=['C37','C07']
   if q['question_id']=='Q05':vq['gold_evidence_ids']=['C37','C07']
   if q['question_id']=='Q04':vq['gold_counterevidence_ids']=['C37','C02']
   if q['question_id']=='Q06':vq['gold_counterevidence_ids']=['C37','C12']
   if q['question_id']=='Q07':vq['gold_counterevidence_ids']=['C37','C10','C04']
   reason='Добавлено одно несовместимое утверждение. Для Q11/Q12 конфликт касается осведомлённости Веры, а не раскрытия искомого факта: insufficient сохраняется.'
  variant=dict(variant_id=q['question_id']+'_'+kind,kind=kind,question=vq,documents=vd,claims=vc,original_document_ids=[d['document_id'] for d in docs],modified_document_ids=[d['document_id'] for d in vd],changed_claim_ids=changed,applicable=applicable,review={'status':'reviewed_by_ai_author','human_review':False,'note':reason or 'Исходный замороженный эталон.'})
  all_variants.append(variant)
(D/'variants.jsonl').write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in all_variants),encoding='utf8')
print('Variants:',len(all_variants),'applicable:',sum(v['applicable'] for v in all_variants))
