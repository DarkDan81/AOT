import hashlib
import json
from lab04_prompt_program.extract import ROOT,validate

def test_frozen_fifty_complete_reviews_and_evidence():
    blob=(ROOT/'data/reviews.jsonl').read_bytes()
    assert hashlib.sha256(blob).hexdigest()==(ROOT/'data/gold.sha256').read_text().strip()
    reviews=[json.loads(line) for line in blob.decode('utf-8').splitlines()]
    assert len(reviews)==50 and len({r['id'] for r in reviews})==50
    for row in reviews:validate(json.dumps(row['gold']),row['text'],grounded=True)

def test_selection_is_complete_review_not_isolated_sentence():
    upstream=[json.loads(line) for line in (ROOT/'data/source.json').read_text(encoding='utf-8').splitlines()]
    selected=[json.loads(line) for line in (ROOT/'data/reviews.jsonl').read_text(encoding='utf-8').splitlines()]
    for row in selected:
        sentences=sorted((s for s in upstream if s['file_name']==row['id']),key=lambda s:s['sentence_id'])
        assert row['text']==''.join(s['text'] for s in sentences)
