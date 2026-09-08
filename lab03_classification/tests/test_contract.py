import hashlib
import json
from collections import Counter
import pandas as pd
import pytest
from lab03_classification.run import HERE,Sentiment

def test_stress_frozen_balanced_categories_unique():
    blob=(HERE/'data/stress.csv').read_bytes()
    assert hashlib.sha256(blob).hexdigest()==(HERE/'data/stress.sha256').read_text().strip()
    data=pd.read_csv(HERE/'data/stress.csv')
    assert len(data)==100 and data.id.nunique()==100 and data.text.nunique()==100
    assert sorted(data.category.value_counts().tolist())==[20]*5
    assert set(data.label)=={'positive','negative','neutral'}

def test_no_train_validation_test_text_leakage():
    splits={k:pd.read_csv(HERE/f'data/{k}.csv') for k in ['train','validation','test']}
    assert [len(splits[k]) for k in splits]==[3000,500,500]
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not set(splits[a].text)&set(splits[b].text)
    assert len(set().union(*(set(d.id) for d in splits.values())))==4000

@pytest.mark.parametrize('bad',[{}, {'label':'skip'}, {'label':'positive','reason':'x'}, {'label':1}])
def test_schema_rejects_invalid(bad):
    with pytest.raises(ValueError):Sentiment.model_validate(bad)

def test_few_shot_six_training_examples():
    prompt=(HERE/'prompts/few_shot.txt').read_text(encoding='utf-8')
    examples=prompt.split('Текст: ')[1:]
    assert len(examples)==6
    train=pd.read_csv(HERE/'data/train.csv')
    labels=[]
    for example in examples:
        text,answer=example.split('\nОтвет: ',1)
        label=json.loads(answer.strip())['label'];labels.append(label)
        found=train[train.text.str.strip()==text.strip()]
        assert len(found)>=1 and label in set(found.label)
    assert Counter(labels)=={'positive':2,'negative':2,'neutral':2}
