import json
from pathlib import Path
from pydantic import BaseModel, ConfigDict, StrictStr

ROOT=Path(__file__).parent
FIELDS=('drug','indication','positive_effect','adverse_reaction')

class Extraction(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    drug:list[StrictStr]
    indication:list[StrictStr]
    positive_effect:list[StrictStr]
    adverse_reaction:list[StrictStr]
    evidence:list[StrictStr]

def validate(content,text,grounded=False):
    value=Extraction.model_validate(json.loads(content))
    for quote in value.evidence:
        if not quote or quote not in text:
            raise ValueError(f'evidence отсутствует дословно в тексте: {quote!r}')
    if grounded:
        for field in FIELDS:
            for item in getattr(value,field):
                if not item or not any(item in quote for quote in value.evidence):
                    raise ValueError(f'{field}: извлечение {item!r} не покрыто evidence')
    return value

def extract(text,version='version_2',complete=None):
    if complete is None:
        from common.llm import complete
    messages=[{'role':'system','content':(ROOT/'prompts'/f'{version}.txt').read_text(encoding='utf-8')},
              {'role':'user','content':text}]
    attempts=[]
    for attempt in range(2):
        response=complete(messages,schema=Extraction,max_tokens=2048)
        entry=dict(response)
        try:
            parsed=validate(response['content'],text,grounded=version=='version_2')
            entry['error']=None
            attempts.append(entry)
            return {'parsed':parsed.model_dump(),'valid':True,'attempts':attempts}
        except (ValueError,TypeError) as error:
            entry['error']=str(error)
            attempts.append(entry)
            if attempt==0:
                messages += [{'role':'assistant','content':response['content']},
                             {'role':'user','content':'Ответ невалиден. Исправь ровно один раз. Ошибка: '+str(error)}]
    return {'parsed':None,'valid':False,'attempts':attempts}
