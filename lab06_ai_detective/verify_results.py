"""Integrity audit of the completed experiment; no network requests."""
import json,hashlib,pathlib,collections
P=pathlib.Path(__file__).parent
read=lambda p:[json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines()]
rows=read(P/'results/responses.jsonl');variants=read(P/'data/variants.jsonl')
expected={v['variant_id']+'_'+m for v in variants if v['applicable'] for m in 'ABC'}
assert len(rows)==len(expected)==120
assert {r['run_id'] for r in rows}==expected
lookup={r['run_id']:r for r in rows}
for v in variants:
 if not v['applicable']:continue
 for mode in 'ABC':
  r=lookup[v['variant_id']+'_'+mode]
  assert r['gold_status']==v['question']['gold_status']
  context=(P/r['context_file']).read_text(encoding='utf8')
  assert hashlib.sha256(context.encode()).hexdigest()==r['context_sha256']
  assert r['context_document_order']==[d['document_id'] for d in v['documents']]
 b,c=lookup[v['variant_id']+'_B'],lookup[v['variant_id']+'_C']
 assert set(b['context_claim_ids'])==set(c['context_claim_ids'])
 if v['kind']=='permutation':
  for mode in 'BC':assert lookup[v['variant_id']+'_'+mode]['context_sha256']==lookup[v['question']['question_id']+'_baseline_'+mode]['context_sha256']
frozen=json.loads((P/'data/gold_freeze.json').read_text(encoding='utf8'))
for name,digest in frozen['sha256'].items():assert hashlib.sha256((P/'data'/name).read_bytes()).hexdigest()==digest
transports=[r['transport'] for r in rows if isinstance(r.get('transport'),dict)]
models={t['model'] for t in transports};assert len(models)==1
params={json.dumps(t['parameters'],sort_keys=True) for t in transports};assert len(params)==1
request_ids=[t['request_id'] for t in transports];assert len(request_ids)==len(set(request_ids))
summary=dict(n_requests=len(rows),n_transport_responses=len(transports),n_valid=sum(r['valid'] for r in rows),models=list(models),parameters=json.loads(next(iter(params))),all_context_hashes_match=True,gold_hashes_unchanged_since_freeze=True,bc_claim_sets_identical=True,permuted_bc_contexts_byte_identical=True,finish_reasons=dict(collections.Counter(t.get('finish_reason') for t in transports)),max_prompt_tokens=max(t['usage']['prompt_tokens'] for t in transports),total_prompt_tokens=sum(t['usage']['prompt_tokens'] for t in transports),total_completion_tokens=sum(t['usage']['completion_tokens'] for t in transports),total_request_seconds=sum(t['elapsed_seconds'] for t in transports))
(P/'results/final_verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(summary,ensure_ascii=False,indent=2))
