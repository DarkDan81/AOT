import json, re
import pandas as pd
from lab05_mini_rag.retrieve import ROOT, read
def norm(s):return re.sub(r'\W+',' ',(s or '').lower()).strip()
def main():
    gold={x['question_id']:x for x in read(ROOT/'data/train.jsonl')+read(ROOT/'evaluation_private/test_gold.jsonl')}
    rows=[]
    for r in read(ROOT/'results/predictions.jsonl'):
        g=gold[r['question_id']];p=r.get('prediction') or {'answer':None,'answerable':False,'evidence_ids':[]}
        a,b=set(p['evidence_ids']),set(g['evidence_ids']); hit=len(a&b)
        precision=hit/len(a) if a else (1. if not b else 0.)
        recall=hit/len(b) if b else (1. if not a else 0.)
        expected=norm(g['answer']);actual=norm(p['answer'])
        # Values, names and e-mail addresses have deterministic gold; token inclusion permits short framing.
        correct=(all((' '+norm(v)+' ') in (' '+actual+' ') for v in g['answer'].split(', ')) if g['answerable'] else not p['answerable'])
        correct=bool(correct and r['valid'] and p['answerable']==g['answerable'])
        usage=r.get('usage') or {}
        rows.append({'question_id':r['question_id'],'split':g['split'],'mode':r['mode'],'valid':r['valid'],'answer_correct':correct,'answer_exact':expected==actual and r['valid'],'answerable_correct':r['valid'] and p['answerable']==g['answerable'],'gold_answerable':g['answerable'],'pred_answerable':p['answerable'],'evidence_precision':precision,'evidence_recall':recall,'evidence_f1':2*precision*recall/(precision+recall) if precision+recall else 0,'all_gold_in_context':b.issubset(r['context_ids']),'seconds':r.get('elapsed_seconds',0),'input_tokens':usage.get('prompt_tokens',0),'output_tokens':usage.get('completion_tokens',0)})
    df=pd.DataFrame(rows);df.to_csv(ROOT/'results/scores.csv',index=False)
    metrics=df.groupby(['split','mode']).agg(n=('valid','size'),json_validity=('valid','mean'),answer_accuracy=('answer_correct','mean'),answer_exact=('answer_exact','mean'),answerable_accuracy=('answerable_correct','mean'),evidence_precision=('evidence_precision','mean'),evidence_recall=('evidence_recall','mean'),evidence_f1=('evidence_f1','mean'),seconds_mean=('seconds','mean'),input_tokens_mean=('input_tokens','mean'),output_tokens_mean=('output_tokens','mean'))
    for (split,mode),group in df.groupby(['split','mode']):
        valid=group['valid'];g=group['gold_answerable'];p=group['pred_answerable']
        tp=int((g&p&valid).sum());fp=int((~g&p&valid).sum());fn=int((g&(~p|~valid)).sum())
        tn=int((~g&~p&valid).sum())
        metrics.loc[(split,mode),'answerable_precision']=tp/(tp+fp) if tp+fp else 0
        metrics.loc[(split,mode),'answerable_recall']=tp/(tp+fn) if tp+fn else 0
        metrics.loc[(split,mode),'refusal_recall']=tn/int((~g).sum())
        metrics.loc[(split,mode),'evidence_f1_answerable_only']=group.loc[g,'evidence_f1'].mean()
    metrics.to_csv(ROOT/'results/metrics.csv')
    confusion=df.groupby(['split','mode','gold_answerable','pred_answerable']).size().reset_index(name='n');confusion.to_csv(ROOT/'results/answerable_confusion.csv',index=False)
    candidates={
        'retrieval':df[(df['mode']=='bm25') & df['gold_answerable'] & ~df['all_gold_in_context']],
        'llm_with_evidence':df[df['gold_answerable'] & df['all_gold_in_context'] & ~df['answer_correct']],
        'answerability':df[~df['answerable_correct']],
    }
    original={(r['question_id'],r['mode']):r for r in read(ROOT/'results/predictions.jsonl')}
    analysis=['# Разбор ошибок\n','Выборка ниже содержит только наблюдаемые ошибки. Категории могут пересекаться; один случай не становится двумя независимыми наблюдениями.\n']
    for cat,frame in candidates.items():
        analysis.append(f'## {cat}: всего {len(frame)}, показано {min(5,len(frame))}\n')
        for _,r in frame.head(5).iterrows():
            g=gold[r.question_id];raw=original[(r.question_id,r['mode'])]
            analysis.append(f"### {r.question_id}, {r['mode']}\nВопрос: {g['question']}\n\nЭталон: {g['answer']}; доказательства {g['evidence_ids']}. Передано: {raw['context_ids']}. Ответ модели: {json.dumps(raw.get('prediction'),ensure_ascii=False)}.\n")
            reason={'retrieval':'Вопрос требует шести самостоятельных разрешений, а top-5 физически содержит не более пяти доказательных абзацев. Полный ответ по такому контексту невозможен; корректный отказ генератора всё равно является ошибкой всей RAG-системы.', 'llm_with_evidence':'Все эталонные абзацы присутствовали, поэтому отсутствие или неверное значение нельзя объяснить потерей retrieval. Следует проверять интерпретацию условия и полноту ответа генератора.', 'answerability':'Решение о достаточности контекста расходится с эталоном полного документа. Если доказательство потеряно при retrieval, отказ локально правилен, но относительно исходного документа это ложноотрицательное решение.'}[cat]
            analysis.append(reason+'\n')
        if len(frame)<5:analysis.append(f'Требуемые пять реальных ошибок этой категории не получены: недостаёт {5-len(frame)}. Успешные ответы не переименовываются в ошибки.\n')
    (ROOT/'results/error_analysis.md').write_text('\n'.join(analysis),encoding='utf-8')
    print(metrics.to_string())
if __name__=='__main__':main()
