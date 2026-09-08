"""Exploratory challenge, frozen separately; never changes the main 100 questions."""
import hashlib,json,sys
from pathlib import Path
from lab05_mini_rag.retrieve import ROOT,read,top5
from lab05_mini_rag.generate import Answer,validate
from common.llm import complete

def freeze():
    cases=[
      ('conflict','Каков единственный действующий лимит копирования? Укажите одно число.', ['Редакция А: действующий лимит копирования составляет 17 страниц.','Редакция Б: действующий лимит копирования составляет 29 страниц.','Редакции имеют одинаковую дату и силу; приоритет не установлен.'],None,[]),
      ('conflict','В каком единственном зале сейчас находится резервный ключ?', ['Акт первого хранителя: резервный ключ находится в зале Север.','Акт второго хранителя: резервный ключ находится в зале Юг.','Акты составлены одновременно; выбрать более надёжного хранителя по материалу невозможно.'],None,[]),
      ('conflict','Кто является единственным действующим директором архива?', ['Приказ первого учредителя назначает единственным директором Анну.','Приказ второго учредителя назначает единственным директором Бориса.','Оба учредителя равноправны; порядок разрешения несовместимых приказов не установлен.'],None,[]),
      ('conflict','Во сколько точно открыли архив? Нужна одна отметка.', ['Журнал А указывает фактическое открытие в 09:10.','Журнал Б указывает фактическое открытие в 09:40.','Журналы равнозначны; часы не сверялись и приоритет ни одному журналу не присвоен.'],None,[]),
      ('conflict','Какой единственный срок уничтожения журнала установлен?', ['Подписанный регламент отдела А устанавливает уничтожение через 90 дней.','Подписанный регламент отдела Б устанавливает уничтожение через 120 дней.','Оба регламента одновременно распространяются на один журнал; правило выбора отсутствует.'],None,[]),
      ('old_rule','Сколько страниц разрешено копировать сейчас?', ['Старый лимит составлял 50 страниц.','Старый лимит отменён. Новый лимит утверждён отдельным неопубликованным приказом.'],None,[]),
      ('quotation','Кто отключил электричество?', ['В учебной сценке есть реплика: «Я, Антон, отключил электричество».','Реплика описывает только вымышленную сценку; реального отключения в архиве не зарегистрировано.'],None,[]),
      ('false_premise','Во сколько Ирина похитила резервный ключ?', ['Ирина не похищала резервный ключ. Ключ весь день оставался в опечатанном шкафу.'],None,[]),
      ('counterfactual','В какой зал фактически эвакуировали посетителей?', ['Если бы произошла авария, посетителей эвакуировали бы в зал Восток.','Аварии не было, эвакуацию не проводили.'],None,[]),
      ('proposal','Какой бюджет архива утверждён?', ['Предложено утвердить бюджет 700000 рублей.','Предложение ещё не рассмотрено; сумма утверждённого бюджета отсутствует в документе.'],None,[]),
      ('exception','Сколько страниц может скопировать преподаватель с учебным направлением без отдельного согласования?', ['Обычный лимит — 20 страниц. Для преподавателей лимит увеличивается на 10 страниц.','Увеличение не действует при учебном направлении: держатель такого направления получает ровно обычный лимит.'], '20',[1,2]),
      ('negation','Кто из перечисленных сотрудников не освобождён от проверки? Укажите имя.', ['От проверки освобождены все сотрудники, кроме Ирины.','Антон и Борис являются сотрудниками. Ирина является сотрудником.'], 'Ирина',[1]),
      ('superseded','Какой срок хранения действует после всех изменений?', ['Первоначальный срок — 90 дней. Приказ Б заменил его на 120 дней.','Приказ В отменил приказ Б и восстановил первоначальный срок.'], '90',[1,2]),
      ('cross_reference','В каком шкафу следует хранить журнал согласно действующему правилу?', ['При действующем разрешении журнал хранят в шкафу С-18, иначе — в шкафу С-29.','Разрешение было выдано, затем отозвано; новое не выдавали.'], 'С-29',[1,2]),
      ('exception','Сколько разрешений действительно нужно для этапа Е?', ['Для каждого этапа нужно 3 разрешения, кроме этапа Е.','Этап Е требует на одно разрешение меньше обычного числа; данное исключение не отменено.'], '2',[1,2]),
    ]
    records=[]
    for i,(kind,q,ps,a,ev) in enumerate(cases):
        paragraphs=[{'id':j+1,'text':t} for j,t in enumerate(ps)]
        paragraphs += [{'id':j+1,'text':f'Фоновая запись {j+1}: в читальном зале хранят бумагу и карандаши. Запись не устанавливает правила или события из вопроса.'} for j in range(len(ps),24)]
        records.append({'question_id':f'CH{i+1:02d}','kind':kind,'question':q,'document':{'document_id':f'CHD{i+1:02d}','paragraphs':paragraphs},'answer':a,'answerable':a is not None,'evidence_ids':ev})
    dest=ROOT/'challenge';dest.mkdir(exist_ok=True)
    for name,rows in [('gold.jsonl',records),('inputs.jsonl',[{k:v for k,v in r.items() if k not in ('answer','answerable','evidence_ids')} for r in records])]:
        (dest/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
    (dest/'freeze.json').write_text(json.dumps({f:hashlib.sha256((dest/f).read_bytes()).hexdigest() for f in ['gold.jsonl','inputs.jsonl']},indent=2),encoding='utf8')
    print('challenge frozen: 15 cases')

def run(dest=None):
    dest=dest or ROOT/'challenge';out=dest/'predictions.jsonl';done={(r['question_id'],r['mode']) for r in read(out)} if out.exists() else set()
    prompt=(ROOT/'prompt.txt').read_text(encoding='utf8')
    for q in read(dest/'inputs.jsonl'):
        for mode in ['full','bm25']:
            if (q['question_id'],mode) in done:continue
            ps=q['document']['paragraphs'] if mode=='full' else top5(q['document'],q['question'])
            context='\n\n'.join(f"[{p['id']}] {p['text']}" for p in ps)
            r=complete([{'role':'system','content':prompt},{'role':'user','content':f"Вопрос: {q['question']}\nКонтекст:\n{context}"}],schema=Answer.model_json_schema(),max_tokens=700)
            row={'question_id':q['question_id'],'mode':mode,'context_ids':[p['id'] for p in ps],**r}
            try:row.update(prediction=validate(r['content'],set(row['context_ids'])),valid=True)
            except Exception as e:row.update(prediction=None,valid=False,error=str(e))
            with out.open('a',encoding='utf8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            print(q['question_id'],mode,flush=True)
if __name__=='__main__':
    freeze() if sys.argv[1]=='freeze' else run()
