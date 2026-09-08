def evidence_metrics(pred,gold):
    p,g=set(pred),set(gold); hit=len(p&g)
    precision=hit/len(p) if p else float(not g)
    recall=hit/len(g) if g else float(not p)
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.
    return precision,recall,f1

def score(answer,q):
    p,r,f=evidence_metrics(answer['evidence_ids'],q['gold_evidence_ids'])
    counter=evidence_metrics(answer['counterevidence_ids'],q['gold_counterevidence_ids'])[1] if q['gold_counterevidence_ids'] else None
    return dict(status_correct=int(answer['status']==q['gold_status']),evidence_precision=p,evidence_recall=r,evidence_f1=f,counterevidence_recall=counter,
    both_correct=int(answer['status']=='both') if q['gold_status']=='both' else None,
    insufficient_correct=int(answer['status']=='insufficient') if q['gold_status']=='insufficient' else None,
    confident_when_insufficient=int(answer['confidence']>=.8 and answer['status']!='insufficient') if q['gold_status']=='insufficient' else None)

