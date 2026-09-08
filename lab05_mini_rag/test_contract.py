import json
import pytest
from lab05_mini_rag.generate import validate
from lab05_mini_rag.retrieve import ROOT, read, top5

def test_dataset_contract():
    docs=read(ROOT/'data/documents.jsonl');qs=read(ROOT/'data/questions.jsonl')
    assert len(docs)==30 and all(20<=len(d['paragraphs'])<=50 for d in docs)
    assert len(qs)==100 and len({(q['document_id'],q['question']) for q in qs})==100
    train=read(ROOT/'data/train.jsonl');test=read(ROOT/'evaluation_private/test_gold.jsonl')
    assert sum(not q['answerable'] for q in train+test)>=20
    assert not {q['document_id'] for q in train}&{q['document_id'] for q in test}
    assert all('answer' not in q for q in read(ROOT/'data/test.jsonl'))

@pytest.mark.parametrize('payload',[
    {'answer':'x','answerable':True,'evidence_ids':[99]},
    {'answer':'x','answerable':False,'evidence_ids':[]},
    {'answer':'x','answerable':True,'evidence_ids':[]},
    {'answer':'x','answerable':True,'evidence_ids':[1],'extra':0},
    {'answer':'x','answerable':True,'evidence_ids':['1']},
])
def test_reject_invalid(payload):
    with pytest.raises(ValueError):validate(json.dumps(payload),{1,2})

def test_refusal_and_answer():
    assert validate('{"answer":null,"answerable":false,"evidence_ids":[]}',{1})['answer'] is None
    assert validate('{"answer":"x","answerable":true,"evidence_ids":[1]}',{1})['answer']=='x'

def test_bm25_six_evidence_ceiling():
    docs={d['document_id']:d for d in read(ROOT/'data/documents.jsonl')}
    q=next(q for q in read(ROOT/'data/train.jsonl') if len(q['evidence_ids'])==6)
    result=top5(docs[q['document_id']],q['question'])
    assert len(result)==5 and not set(q['evidence_ids']).issubset(p['id'] for p in result)
