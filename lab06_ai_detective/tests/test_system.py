import json,sys
from pathlib import Path
from itertools import product
from unittest.mock import patch
import pytest
P=Path(__file__).parents[1];sys.path.insert(0,str(P));sys.path.insert(0,str(P.parent))
from lab06_ai_detective.src.padic import code,distance
from lab06_ai_detective.src.retrieval import retrieve
from lab06_ai_detective.src.context_builder import build
from lab06_ai_detective.src.schemas import validate

def read(name):return [json.loads(x) for x in (P/'data'/name).read_text(encoding='utf-8-sig').splitlines()]
DOCS=read('documents.jsonl');CLAIMS=read('claims.jsonl');QUESTIONS=read('questions.jsonl')
def raw(**changes):
 d=dict(question_id='Q01',status='entailed',answer='18:12',evidence_ids=['C08'],counterevidence_ids=[],confidence=.8,explanation='C08 states the time.')
 d.update(changes);return json.dumps(d)
def test_same_path():assert distance([1,2,1],[1,2,1])==0
@pytest.mark.parametrize('other,expected',[([2,0,0],1),([0,2,0],1/3),([0,0,2],1/9)])
def test_each_level(other,expected):assert distance([0,0,0],other)==expected
def test_code():
 assert code([1,2,1])==16
 assert distance([0,0,0],[0,0,1])==1/9
def test_ultrametric_exhaustive():
 paths=list(product(range(3),repeat=3))
 for a,b,c in product(paths,repeat=3):assert distance(a,c)<=max(distance(a,b),distance(b,c))
def test_invalid_json():
 with pytest.raises(ValueError):validate('```json {} ```',['C08'],'Q01')
def test_nonexistent_id():
 with pytest.raises(ValueError):validate(raw(evidence_ids=['C99']),['C08'],'Q01')
def test_both_without_counter():
 with pytest.raises(ValueError):validate(raw(status='both'),['C08'],'Q01')
def test_insufficient_with_invented_answer():
 with pytest.raises(ValueError):validate(raw(status='insufficient',answer='Invented motive'),['C08'],'Q01')
def test_llm_unavailable():
 from lab06_ai_detective.src.llm import ask
 with patch('lab06_ai_detective.src.llm.complete',side_effect=ConnectionError('server unavailable')):
  with pytest.raises(ConnectionError):ask(QUESTIONS[0],'context',['C08'])
def test_bm25_maximum():assert len(retrieve('Лада',CLAIMS,100))==15
def test_bc_same_ids_every_variant():
 for v in read('variants.jsonl'):
  selected=retrieve(v['question']['question'],v['claims'])
  b,bi=build('B',v['question']['question'],v['documents'],v['claims'],selected)
  c,ci=build('C',v['question']['question'],v['documents'],v['claims'],selected)
  assert set(bi)==set(ci)
  assert all(b.count('"claim_id": "'+i+'"')==c.count('"claim_id": "'+i+'"')==1 for i in bi)
def test_evidence_verbatim_unique_ids():
 docs={d['document_id']:d for d in DOCS}
 assert len({c['claim_id'] for c in CLAIMS})==36
 for c in CLAIMS:assert c['evidence'] in docs[c['document_id']]['text']
 assert 5000<=sum(len(d['text'].split()) for d in DOCS)<=8000
 assert len(DOCS)==9
 assert '????' not in ''.join(d['text'] for d in DOCS)
def test_deletion_really_removes_full_context():
 for v in read('variants.jsonl'):
  if v['kind']!='deletion' or not v['applicable']:continue
  removed=next(c for c in CLAIMS if c['claim_id']==v['changed_claim_ids'][0])
  text,ids=build('A',v['question']['question'],v['documents'],v['claims'])
  assert removed['evidence'] not in text
  assert removed['claim_id'] not in ids
  assert v['question']['gold_status']=='insufficient'
def test_permutation_one_factor():
 for v in read('variants.jsonl'):
  if v['kind']=='permutation':
   assert sorted(v['documents'],key=lambda d:d['document_id'])==DOCS
   assert v['claims']==CLAIMS
def test_contradiction_one_factor():
 for v in read('variants.jsonl'):
  if v['kind']=='contradiction':
   assert v['documents'][:-1]==DOCS
   assert v['claims'][:-1]==CLAIMS
   assert len(v['claims'])==37


def test_insufficient_fabricated_explanation():
 with pytest.raises(ValueError):validate(raw(status='insufficient',answer=None,explanation='Илья удалил файл из мести.'),['C08'],'Q01')

def test_raw_invalid_response_retained():
 from lab06_ai_detective.src.llm import ask,ResponseValidationError
 response={'content':'not json','usage':{'prompt_tokens':10}}
 with patch('lab06_ai_detective.src.llm.complete',return_value=response):
  with pytest.raises(ResponseValidationError) as e:ask(QUESTIONS[0],'context',['C08'])
 assert e.value.response==response


@pytest.mark.parametrize('http_response',[False,True])
def test_network_error_serializable(http_response):
 import requests
 from lab06_ai_detective.run_experiment import error_record
 if http_response:
  response=requests.Response();response.status_code=503;response._content=b'{"error":"unavailable"}'
  error=requests.HTTPError('503',response=response)
 else:error=requests.ConnectionError('connection refused')
 row=error_record(error,1.0)
 assert row['valid'] is False
 assert 'transport' not in row
 assert json.loads(json.dumps(row))['error']
 if http_response:assert row['http_status']==503 and row['http_body']=='{"error":"unavailable"}'
