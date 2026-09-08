"""Separate exploratory metrics and five distinct observed full-context errors."""
import json
import pandas as pd
from lab05_mini_rag.retrieve import ROOT,read
from lab05_mini_rag.evaluate import norm

def main():
    all_rows=[];analyses=[]
    for stage in ['challenge','challenge_replication']:
        path=ROOT/stage;gold={r['question_id']:r for r in read(path/'gold.jsonl')}
        for r in read(path/'predictions.jsonl'):
            g=gold[r['question_id']];p=r.get('prediction') or {'answer':None,'answerable':False,'evidence_ids':[]}
            correct=bool(r['valid'] and p['answerable']==g['answerable'] and (not g['answerable'] or (' '+norm(g['answer'])+' ') in (' '+norm(p['answer'])+' ')))
            needed={x['id'] for x in g['document']['paragraphs'] if not x['text'].startswith('Фоновая запись')}
            all_rows.append({'stage':stage,'question_id':r['question_id'],'mode':r['mode'],'valid':r['valid'],'answer_correct':correct,'answerable_correct':r['valid'] and p['answerable']==g['answerable'],'all_relevant_in_context':needed.issubset(r['context_ids']),'seconds':r['elapsed_seconds'],'input_tokens':r['usage']['prompt_tokens'],'output_tokens':r['usage']['completion_tokens']})
            if r['mode']=='full' and not correct:
                analyses.append(f"### {r['question_id']}, {stage}, full\n\nВопрос: {g['question']}\n\nАбзац 1: {g['document']['paragraphs'][0]['text']}\n\nАбзац 2: {g['document']['paragraphs'][1]['text']}\n\nЭталон: null, answerable=false. Модель: {json.dumps(p,ensure_ascii=False)}. Оба абзаца присутствовали среди всех 24 переданных абзацев.\n\nПричина: генератор перенёс исполнителя из учебной сценки на событие архива. Дословное существование цитаты не доказывает реальное совершение действия. Поэтому ссылка [1] проходит структурную проверку, но семантически не подтверждает искомое событие. Нужна проверка области действия цитаты, а не только существования ID.\n")
    df=pd.DataFrame(all_rows)
    df['shared_gpu_workload']=df['stage'].eq('challenge_replication')
    df.to_csv(ROOT/'results/challenge_scores.csv',index=False)
    metrics=df.groupby(['stage','mode']).agg(n=('valid','size'),json_validity=('valid','mean'),answer_accuracy=('answer_correct','mean'),answerable_accuracy=('answerable_correct','mean'),seconds_mean=('seconds','mean'),input_tokens_mean=('input_tokens','mean'),output_tokens_mean=('output_tokens','mean'))
    metrics['timing_comparable']=False
    metrics.to_csv(ROOT/'results/challenge_metrics.csv')
    header='\n## Пять наблюдаемых ошибок LLM при правильном полном контексте\n\nЭто отдельные exploratory-наборы; они не входят в метрики основных 100 вопросов. CH07 выявлен в наборе 15 новых случаев; QR01–QR04 целенаправленно проверяют тот же сбой после наблюдения CH07. Четыре варианта заморожены до своих вызовов. Это пять разных вопросов, но один механизм ошибки, а не пять независимых открытий.\n\n'
    text=header+'\n'.join(analyses)
    (ROOT/'results/challenge_error_analysis.md').write_text(text,encoding='utf8')
    base=ROOT/'results/error_analysis.md';s=base.read_text(encoding='utf8').split('\n## Пять наблюдаемых ошибок LLM')[0]
    base.write_text(s+text,encoding='utf8')
    print(metrics.to_string());print('Distinct full-context errors:',len(analyses))
if __name__=='__main__':main()
