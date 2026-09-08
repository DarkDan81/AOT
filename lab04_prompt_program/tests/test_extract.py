import json
import pytest
from lab04_prompt_program.extract import extract,validate

def obj(**updates):
    value={k:[] for k in ('drug','indication','positive_effect','adverse_reaction','evidence')}
    value.update(updates)
    return json.dumps(value,ensure_ascii=False)

class Fake:
    def __init__(self,*answers):self.answers=list(answers);self.messages=[]
    def __call__(self,messages,**kwargs):
        self.messages.append(list(messages))
        return {'content':self.answers.pop(0),'usage':{},'elapsed_seconds':0,'model':'test-double'}

@pytest.mark.parametrize('text,answer',[
    ('',obj()),
    ('Средство помогло.',obj(positive_effect=['помогло'],evidence=['Средство помогло.'])),
    ('Пила А и Б.',obj(drug=['А','Б'],evidence=['Пила А и Б.'])),
    ('А не помог.',obj(drug=['А'],evidence=['А не помог.'])),
    ('А может помочь.',obj(drug=['А'],evidence=['А может помочь.'])),
    ('У ребенка от А появилась сыпь.',obj(drug=['А'],adverse_reaction=['сыпь'],evidence=['У ребенка от А появилась сыпь.'])),
    ('В отзыве: «А вызвал сыпь». У меня сыпи нет.',obj(drug=['А'],evidence=['А вызвал сыпь'])),
    ('Пила Асперин.',obj(drug=['Асперин'],evidence=['Пила Асперин.'])),
],ids=['empty','no-drug','multiple-drugs','negation','hypothesis','other-patient','quotation','typo'])
def test_interface_cases(text,answer):
    # Unit tests verify accepted contracts, not the semantic ability of a mock LLM.
    fake=Fake(answer)
    result=extract(text,complete=fake)
    assert result['valid'] and result['parsed']==json.loads(answer)
    assert len(fake.messages)==1

def test_extra_field_repaired():
    fake=Fake(obj(extra=[]),obj())
    result=extract('',complete=fake)
    assert result['valid'] and len(fake.messages)==2
    assert 'extra' in fake.messages[1][-1]['content']

def test_fabricated_evidence_repaired():
    fake=Fake(obj(evidence=['выдумка']),obj())
    assert extract('исходный текст',complete=fake)['valid']
    assert 'выдумка' in fake.messages[1][-1]['content']

def test_exactly_one_retry_even_after_second_failure():
    fake=Fake('not json','not json')
    result=extract('текст',complete=fake)
    assert not result['valid'] and len(result['attempts'])==2

@pytest.mark.parametrize('bad',[obj(drug=[42]),obj(drug='А'),'{}','```json\n{}\n```'])
def test_strict_schema(bad):
    with pytest.raises(ValueError):validate(bad,'А')

def test_extraction_must_be_covered_by_evidence():
    with pytest.raises(ValueError):validate(obj(drug=['А'],evidence=['Б']),'А Б',grounded=True)

def test_empty_quote_not_proof():
    with pytest.raises(ValueError):validate(obj(evidence=['']),'')

def test_case_sensitive_evidence():
    with pytest.raises(ValueError):validate(obj(evidence=['лекарство']),'Лекарство')
