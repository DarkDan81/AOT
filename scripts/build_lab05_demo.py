"""Build the sequential, offline-first lab 5 defense notebook."""
from pathlib import Path
import nbformat as n

cells=[]
def md(s): cells.append(n.v4.new_markdown_cell(s))
def code(s): cells.append(n.v4.new_code_cell(s.strip()))

md('''# ЛР5 — Мини-RAG с доказательствами
Дущенко Даниил Александрович и Коваленко Евгений Юрьевич, К3341.

В работе сравниваются два способа формирования контекста: полный документ и пять абзацев, найденных BM25. Выполняются поиск доказательств, проверка структуры ответов и оценка сохранённых результатов эксперимента.

Цель: сравнить полный документ и BM25 top-5, отделить ошибки поиска от ошибок генератора.
''')
md('## 0. Окружение\nИспользуются Python, pandas, matplotlib, rank-bm25, razdel и Pydantic. Параметры окружения и зависимости зафиксированы в репозитории.')
code('''
from pathlib import Path
import sys, json, hashlib, tempfile, shutil, io, contextlib
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

REPO = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / 'lab05_mini_rag/data/documents.jsonl').exists()), None)
assert REPO is not None, 'Корень репозитория AOT не найден'
sys.path.insert(0, str(REPO))
LAB = REPO / 'lab05_mini_rag'
from lab05_mini_rag.retrieve import read, top5
from lab05_mini_rag.generate import Answer, validate
plt.rcParams.update({'figure.figsize': (10, 4), 'font.size': 11})
print('Python:', sys.executable)
print('Проект:', REPO)
''')
md('## 1. Данные и стабильные идентификаторы\nЗагружаем уже подготовленный корпус. Эталоны тестовой части используются только при оценке; в запрос модели они не включаются.')
code('''
manifest = json.loads((LAB / 'data/freeze_manifest.json').read_text())
for name, expected in manifest.items():
    assert hashlib.sha256((LAB / name).read_bytes()).hexdigest() == expected, name
docs = {d['document_id']: d for d in read(LAB / 'data/documents.jsonl')}
questions = read(LAB / 'data/questions.jsonl')
train = read(LAB / 'data/train.jsonl')
test_gold = read(LAB / 'evaluation_private/test_gold.jsonl')
gold = {q['question_id']: q for q in train + test_gold}
assert len(docs) == 30 and len(questions) == 100
assert {q['document_id'] for q in train}.isdisjoint({q['document_id'] for q in test_gold})
display(pd.DataFrame([{'Документы': len(docs), 'Вопросы': len(questions),
    'Train': len(train), 'Test': len(test_gold),
    'Без ответа': sum(not q['answerable'] for q in gold.values()),
    'Абзацев в документе': len(next(iter(docs.values()))['paragraphs'])}]))
QUESTION_ID = 'Q005'  # Составной вопрос, требующий шести доказательных абзацев.
q = next(q for q in questions if q['question_id'] == QUESTION_ID)
print(q['question'])
display(pd.DataFrame(docs[q['document_id']]['paragraphs']))
''')
md('## 2. BM25 top-5\nИспользуем существующую функцию `retrieve.top5`: razdel, нижний регистр, без лемматизации. Поиск выполняется внутри документа вопроса.')
code('''
import inspect
from IPython.display import Code
from lab05_mini_rag.retrieve import tokens

# Исходный код функций, используемых при поиске.
display(Code(inspect.getsource(tokens) + '\\n\\n' + inspect.getsource(top5), language='python'))
''')
code('''
selected = top5(docs[q['document_id']], q['question'])
display(pd.DataFrame(selected))
print('Эталонные ID:', gold[QUESTION_ID]['evidence_ids'])
print('Полученные ID:', [p['id'] for p in selected])
''')
md('## 3. Recall до обращения к LLM\nВопросы без доказательных абзацев не входят в среднее recall. Для шестичастного вопроса пять абзацев физически не могут содержать все шесть обязательных посылок.')
code('''
retrieval_rows = []
for item in gold.values():
    found = {p['id'] for p in top5(docs[item['document_id']], item['question'])}
    expected = set(item['evidence_ids'])
    if expected:
        retrieval_rows.append({'split': item['split'], 'question_id': item['question_id'],
            'recall': len(found & expected)/len(expected), 'complete': expected <= found})
retrieval = pd.DataFrame(retrieval_rows)
display(retrieval.groupby('split').agg(n=('recall','size'), recall_at_5=('recall','mean'),
                                      full_evidence=('complete','mean')).round(5))
assert len(retrieval) == 70 and retrieval['complete'].sum() == 60
ax = retrieval.groupby('split')[['recall','complete']].mean().rename(
    columns={'recall':'Средний recall', 'complete':'Все доказательства'}).plot.bar(ylim=(0,1.05), rot=0)
ax.set(title='BM25: средний recall и достаточность контекста', ylabel='Доля', xlabel='Часть корпуса')
plt.tight_layout(); plt.show()
''')
md('## 4. Один промпт для двух режимов\nМодель получает вопрос и контекст. Эталонный ответ не передаётся. Настройки исходного эксперимента: Ministral 3 14B Reasoning, temperature=0, seed=42, max_tokens=700, JSON Schema.')
code("prompt = (LAB / 'prompt.txt').read_text(encoding='utf-8')\nprint(prompt)")
md('## 5. Проверка ответа через Pydantic\nПроверяем строгие типы, согласованность отказа, отсутствие повторов ID и наличие ссылок в переданном контексте. Эти проверки не доказывают истинность ответа.')
code('''
examples = [
    ('Корректный отказ', {'answer': None, 'evidence_ids': [], 'answerable': False}),
    ('Ссылка вне контекста', {'answer': 'Ответ', 'evidence_ids': [999], 'answerable': True}),
    ('Несогласованный отказ', {'answer': 'Ответ', 'evidence_ids': [], 'answerable': False})]
checks = []
for title, response in examples:
    try:
        validate(json.dumps(response), {p['id'] for p in selected})
        result = 'Принят'
    except (ValueError, TypeError) as error:
        result = 'Отклонён: ' + str(error).splitlines()[0]
    checks.append({'Пример': title, 'Результат': result})
display(pd.DataFrame(checks))
''')
md('## 6. Пересчёт метрик и графики\nЗапускаем существующие `evaluate.py` и `evaluate_challenge.py` над временной копией материалов. Исходные журналы остаются неизменными. Основной эксперимент: 200 сохранённых ответов. Дополнительные 38 ответов оцениваются отдельно.')
code('''
from lab05_mini_rag import evaluate, evaluate_challenge
with tempfile.TemporaryDirectory(prefix='aot5_demo_') as directory:
    scratch = Path(directory)
    for folder in ['data', 'evaluation_private', 'results', 'challenge', 'challenge_replication']:
        shutil.copytree(LAB / folder, scratch / folder)
    for module in [evaluate, evaluate_challenge]:
        original_root = module.ROOT
        try:
            module.ROOT = scratch
            with contextlib.redirect_stdout(io.StringIO()):
                module.main()
        finally:
            module.ROOT = original_root
    metrics = pd.read_csv(scratch / 'results/metrics.csv')
    scores = pd.read_csv(scratch / 'results/scores.csv')
    extra_metrics = pd.read_csv(scratch / 'results/challenge_metrics.csv')
assert metrics['n'].sum() == 200 and extra_metrics['n'].sum() == 38
reference = pd.read_csv(LAB / 'results/metrics.csv')
pd.testing.assert_frame_equal(metrics, reference, check_exact=False, rtol=1e-9, atol=1e-9)
columns = ['split','mode','n','answer_accuracy','json_validity','evidence_f1',
           'seconds_mean','input_tokens_mean','output_tokens_mean']
display(metrics[columns].round(3))
print('Пересчитанные метрики совпадают с сохранёнными результатами.')
''')
code('''
test = metrics[metrics['split'] == 'test'].set_index('mode').reindex(['full','bm25'])
fig, axes = plt.subplots(1, 3, figsize=(15,4))
test[['answer_accuracy','evidence_f1']].rename(columns={
    'answer_accuracy':'Полный ответ','evidence_f1':'Evidence F1'}).plot.bar(ax=axes[0], rot=0, ylim=(0,1.1))
axes[0].set_title('Качество на test')
test['input_tokens_mean'].plot.bar(ax=axes[1], rot=0, color=['#4274b3','#df9135'])
axes[1].set_title('Среднее число входных токенов')
test['seconds_mean'].plot.bar(ax=axes[2], rot=0, color=['#4274b3','#df9135'])
axes[2].set_title('Среднее время ответа, с')
for ax in axes: ax.set_xlabel('Режим')
plt.tight_layout(); plt.show()
display(metrics[['split','mode','answerable_accuracy','context_answerable_accuracy','refusal_recall']])
''')
md('''`answerable_accuracy` оценивается относительно полного документа: 1,00 у обоих режимов. `context_answerable_accuracy` оценивает достаточность переданных абзацев: 1,00 у full и 0,90 у BM25. Поэтому правильное определение наличия ответа в исходном документе совместимо с ошибочным ответом по неполному контексту.

Качество полного ответа проверяется по наличию всех эталонных значений с границами токенов. Evidence F1 для двух пустых списков равно 1. Время на графике — исходное измеренное время LLM, а не длительность пересчёта notebook.''')
md('## 7. Пятнадцать ошибок\nПервые десять случаев относятся к основному набору. CH07 и QR01–QR04 — отдельные диагностические эксперименты: в основном наборе full-ошибок не было. Все пять дополнительных full-ошибок воспроизводят один механизм переноса события из сценки в действительность.')
code('''
saved = {(r['question_id'], r['mode']): r for r in read(LAB / 'results/predictions.jsonl')}
error_rows = []
for category, ids in [('Retrieval', ['Q005','Q015','Q025','Q035','Q045']),
                       ('Answerable', ['Q055','Q065','Q075','Q085','Q095'])]:
    for identifier in ids:
        r = saved[(identifier, 'bm25')]
        error_rows.append({'ID': identifier, 'Этап': 'Основной', 'Ошибка': category,
            'Эталон': gold[identifier]['answer'], 'Ответ': r['prediction']['answer'],
            'Эталонные ID': gold[identifier]['evidence_ids'], 'Переданные ID': r['context_ids']})
for stage, ids in [('challenge', ['CH07']), ('challenge_replication', ['QR01','QR02','QR03','QR04'])]:
    stage_gold = {r['question_id']: r for r in read(LAB / stage / 'gold.jsonl')}
    for r in read(LAB / stage / 'predictions.jsonl'):
        if r['mode'] == 'full' and r['question_id'] in ids:
            g = stage_gold[r['question_id']]
            error_rows.append({'ID': r['question_id'], 'Этап': stage, 'Ошибка': 'LLM при полном контексте',
                'Эталон': g['answer'], 'Ответ': r['prediction']['answer'],
                'Эталонные ID': g['evidence_ids'], 'Переданные ID': r['context_ids']})
assert len(error_rows) == 15
display(pd.DataFrame(error_rows))
display(extra_metrics[['stage','mode','n','answer_accuracy','json_validity']].round(3))
''')
code('''
# Подробный пример дополнительной ошибки: вопрос, исходные абзацы и реальный ответ.
g = next(g for g in read(LAB / 'challenge/gold.jsonl') if g['question_id'] == 'CH07')
r = next(r for r in read(LAB / 'challenge/predictions.jsonl') if r['question_id'] == 'CH07' and r['mode'] == 'full')
print(g['question'])
display(pd.DataFrame(g['document']['paragraphs'][:2]))
print('Эталон:', g['answer'], 'answerable:', g['answerable'])
display(r['prediction'])
''')
md('''## 8. Получение новых ответов модели
Используется локальный сервер `http://localhost:1234/v1`, модель `mistralai/ministral-3-14b-reasoning` и контекст 24 576 токенов.

При `RUN_LIVE = True` выполняются два новых запроса для выбранного вопроса: full и BM25. Каждый ответ сохраняется в `demo_outputs/live_responses.jsonl`, включая невалидные ответы. Результаты этого этапа оцениваются отдельно от основного эксперимента.''')
code('''
RUN_LIVE = False
LIVE_QUESTION_ID = 'Q005'
if RUN_LIVE:
    from common.llm import complete, BASE_URL, MODEL
    live_q = next(item for item in questions if item['question_id'] == LIVE_QUESTION_ID)
    output = LAB / 'demo_outputs'
    output.mkdir(exist_ok=True)
    print('Сервер:', BASE_URL, 'Модель:', MODEL)
    for mode in ['full', 'bm25']:
        paragraphs = (docs[live_q['document_id']]['paragraphs'] if mode == 'full'
                      else top5(docs[live_q['document_id']], live_q['question']))
        context = '\\n\\n'.join(f"[{p['id']}] {p['text']}" for p in paragraphs)
        response = complete([{'role':'system','content':prompt},
            {'role':'user','content':f"Вопрос: {live_q['question']}\\nКонтекст:\\n{context}"}],
            schema=Answer.model_json_schema(), max_tokens=700)
        record = {'question_id': LIVE_QUESTION_ID, 'mode': mode, 'context': context,
                  'context_ids': [p['id'] for p in paragraphs], **response}
        try:
            record.update(prediction=validate(response['content'], set(record['context_ids'])), valid=True)
        except (ValueError, TypeError) as error:
            record.update(prediction=None, valid=False, error=str(error))
        with (output / 'live_responses.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\\n')
        print(mode, 'Время:', round(response['elapsed_seconds'], 2), 'с')
        display(record.get('prediction') or record)
else:
    print('Повторная генерация не выполнялась; выше проанализированы сохранённые ответы основного эксперимента.')
''')
md('''## Выводы
На основном корпусе full даёт 100% полных ответов, BM25 — 90%, сокращая вход примерно на 63% и время примерно на 11% на test. Средний recall 0,9762 скрывает потерю обязательной шестой посылки. Структурная валидность JSON и существование ссылок не гарантируют правильной интерпретации текста.

Дополнительные эксперименты показывают ошибки LLM даже при полном контексте; их показатели не смешиваются с основной таблицей. Корпус синтетический, поэтому результаты нельзя автоматически переносить на произвольные документы.

Вклад участников: **Коваленко Е.Ю.** — данные, сегментация, BM25 и поиск; **Дущенко Д.А.** — промпт, генерация, валидация, метрики и ошибки. Отчёт подготовлен совместно.
''')
nb=n.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python (AOT)', 'language':'python','name':'aot'},'language_info':{'name':'python','version':'3.12'}})
target=Path(__file__).resolve().parents[1]/'lab05_mini_rag/lab05_demo.ipynb'
n.write(nb,target)
print(target)
