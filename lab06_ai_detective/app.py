import json,pathlib,sys,hashlib
import pandas as pd
import streamlit as st
P=pathlib.Path(__file__).parent;sys.path.insert(0,str(P));sys.path.insert(0,str(P.parent))
from src.context_builder import build
from src.retrieval import retrieve
from src.padic import branches,code
from src.llm import ask

def read(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]
st.set_page_config(page_title='AI-детектив: Орион',layout='wide')
st.title('AI-детектив: досье «Орион»')
st.caption('Все события вымышлены. Результаты получены одной локальной моделью; независимые роли моделируются ИИ.')
variants=read(P/'data/variants.jsonl');questions=read(P/'data/questions.jsonl')
with st.sidebar:
 st.header('Исследование')
 uploaded=st.file_uploader('Загрузить documents.jsonl',type=['jsonl'])
 q=st.selectbox('Вопрос',questions,format_func=lambda x:x['question_id']+' — '+x['question'])
 mode=st.radio('Режим',['A','B','C'],format_func=lambda x:{'A':'A — полный контекст','B':'B — BM25 top-15','C':'C — p-adic ветви'}[x])
 mutation=st.selectbox('Изменение',['baseline','permutation','deletion','contradiction'],format_func=lambda x:{'baseline':'Исходное досье','permutation':'Переставить документы (42)','deletion':'Удалить необходимую посылку','contradiction':'Добавить подготовленное противоречие'}[x])
 v=next(x for x in variants if x['question']['question_id']==q['question_id'] and x['kind']==mutation)
 manual_remove=None
 if mutation=='deletion':
  baseline=next(x for x in variants if x['question']['question_id']==q['question_id'] and x['kind']=='baseline')
  options=baseline['claims'];default=next((i for i,c in enumerate(options) if c['claim_id'] in v['changed_claim_ids']),0)
  chosen=st.selectbox('Удаляемая посылка',options,index=default,format_func=lambda c:c['claim_id']+' — '+c['text'])
  if not v['applicable'] or chosen['claim_id'] not in v['changed_claim_ids']:manual_remove=chosen
 docs=v['documents'];claims=v['claims'];custom=False
 if manual_remove:
  import copy
  docs=copy.deepcopy(baseline['documents'])
  for d in docs:
   if d['document_id']==manual_remove['document_id']:d['text']=d['text'].replace(manual_remove['evidence'],'')
  claims=[c for c in baseline['claims'] if c['claim_id']!=manual_remove['claim_id']]
  custom=True;st.info('Произвольное удаление: эталон не проверен, автоматическая оценка отключена.')
 if uploaded:
  try:
   docs=[json.loads(x) for x in uploaded.getvalue().decode('utf-8-sig').splitlines() if x.strip()]
   if len({d['document_id'] for d in docs})!=len(docs):raise ValueError('Повтор document_id')
   byid={d['document_id']:d for d in docs}
   claims=[c for c in claims if c['document_id'] in byid and c['evidence'] in byid[c['document_id']]['text']]
   custom=True;st.info('Пользовательское досье: доступны только дословно совпадающие утверждения; сохранённые эталоны не применяются.')
  except Exception as e:st.error(str(e));st.stop()
selected=retrieve(q['question'],claims)
context,ids=build(mode,q['question'],docs,claims,selected)
context_hash=hashlib.sha256(context.encode('utf8')).hexdigest()
a,b=st.columns([3,2])
with a:
 st.subheader(q['question'])
 result_path=P/'results/responses.jsonl';saved=read(result_path) if result_path.exists() else []
 matching=[r for r in saved if r['run_id']==v['variant_id']+'_'+mode]
 if st.button('Запросить локальную модель',disabled=not v['applicable'] and not custom):
  try:
   with st.spinner('Модель анализирует доступный контекст…'):answer,transport=ask(q,context,ids)
   (P/'results/interactive_contexts').mkdir(exist_ok=True)
   (P/'results/interactive_contexts'/f'{context_hash}.txt').write_text(context,encoding='utf8')
   with (P/'results/interactive.jsonl').open('a',encoding='utf8') as handle:
    handle.write(json.dumps({'question':q,'mode':mode,'context_sha256':context_hash,'custom_context':custom,'answer':answer,'transport':transport},ensure_ascii=False)+'\n')
   st.session_state['live']={'key':(v['variant_id'],mode,context_hash),'answer':answer,'transport':transport}
  except Exception as e:st.error(f'Запрос не выполнен: {e}')
 live=st.session_state.get('live',{})
 current=live if live.get('key')==(v['variant_id'],mode,context_hash) else matching[-1] if matching and not custom else None
 if current and current.get('answer'):
  answer=current['answer'];st.metric('Логический статус',answer['status']);st.metric('Confidence',f"{answer['confidence']:.2f}")
  st.write(answer['answer'] or 'Однозначный ответ не дан.');st.write(answer['explanation'])
  for title,key in [('Доказательства','evidence_ids'),('Контрдоказательства','counterevidence_ids')]:
   st.markdown('**'+title+'**')
   for cid in answer[key]:
    found=next((c for c in claims if c['claim_id']==cid),None)
    st.write(cid,found['evidence'] if found else 'Недоступное утверждение')
 elif current:st.error(current.get('error','Невалидный ответ'))
 else:st.info('Для этого режима пока нет сохранённого ответа.')
with b:
 st.subheader('p-adic ветви выбранных утверждений')
 for path,items in branches(selected):
  with st.expander(f'{path}, код {code(path)} — {len(items)} утверждений'):
   for c in items:st.write(c['claim_id'],c['event_time'],c['text'])
with st.expander('Передаваемый контекст и порядок документов'):
 st.write([d['document_id'] for d in docs]);st.code(context,language=None)
 st.download_button('Скачать контекст',context,file_name=f"{v['variant_id']}_{mode}.txt")
with st.expander('Документы выбранного варианта'):
 for d in docs:st.markdown('**'+d['document_id']+' '+d['title']+'**');st.write(d['text'])
st.subheader('Сравнение сохранённых режимов')
comparison=[]
for r in saved:
 if r['variant_id']!=v['variant_id']:continue
 t=r.get('transport',{});answer=r.get('answer',{})
 comparison.append({'Режим':r['mode'],'Статус':answer.get('status','INVALID'),'Evidence F1':r.get('metrics',{}).get('evidence_f1'),'Confidence':answer.get('confidence'),'Токены':t.get('usage',{}).get('total_tokens'),'Секунды':t.get('elapsed_seconds')})
if comparison and not custom:st.dataframe(pd.DataFrame(comparison),hide_index=True)
else:st.info('Сохранённые результаты не найдены или неприменимы к загруженному досье.')
st.caption('Стресс-варианты предварительно проверены агентами. Такая проверка не выдаётся за проверку реальными участниками.')



