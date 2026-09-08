"""Replay acceptance checks against persisted REAL LLM responses, without new API calls."""
import json
import pytest
from lab04_prompt_program.behavioral import CASES
from lab04_prompt_program.extract import ROOT

@pytest.mark.parametrize('case,text,constraints',CASES,ids=[c[0] for c in CASES])
def test_saved_real_model_behavior(case,text,constraints):
    path=ROOT/'results/behavioral.jsonl'
    if not path.exists():pytest.skip('Run python -m lab04_prompt_program.behavioral first')
    rows={r['case']:r for r in map(json.loads,path.read_text(encoding='utf-8').splitlines())}
    assert case in rows
    row=rows[case]
    assert row['text']==text and row['valid']
    pred=row['parsed']
    for key,values in constraints.items():
        if not values:assert pred[key]==[]
        else:assert all(any(value in actual for actual in pred[key]) for value in values)
    assert row['attempts'][0]['model']!='test-double'
