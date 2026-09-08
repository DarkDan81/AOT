import json
from .padic import branches
from .retrieval import retrieve

def line(c):
    # Одинаковые поля в B/C: различаются лишь заголовки ветвей и порядок.
    return json.dumps({k:c[k] for k in ['claim_id','text','document_id','status','path','event_time']},ensure_ascii=False)

def build(mode,question,documents,claims,selected=None):
    if mode=='A':
        text='\n\n'.join(json.dumps(d,ensure_ascii=False) for d in documents)
        text+='\n\nРЕЕСТР УТВЕРЖДЕНИЙ (для ссылок):\n'+'\n'.join(line(c) for c in claims)
        return text,[c['claim_id'] for c in claims]
    selected=selected if selected is not None else retrieve(question,claims)
    if mode=='B':text='\n'.join(line(c) for c in selected)
    elif mode=='C':text='\n\n'.join('ВЕТВЬ '+str(path)+'\n'+'\n'.join(line(c) for c in items) for path,items in branches(selected))
    else:raise ValueError(mode)
    return text,[c['claim_id'] for c in selected]
