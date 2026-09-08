import re
from rank_bm25 import BM25Okapi

def tokens(s):
    return re.findall(r"[а-яёa-z0-9]+",s.lower())

def retrieve(question,claims,k=15):
    if not claims:return []
    index=BM25Okapi([tokens(c['text']) for c in claims])
    scores=index.get_scores(tokens(question))
    order=sorted(range(len(claims)),key=lambda i:(-scores[i],claims[i]['claim_id']))
    return [claims[i] for i in order[:min(k,15)]]
