from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,model_validator

class Answer(BaseModel):
    model_config=ConfigDict(extra='forbid')
    question_id:str
    status:Literal['entailed','contradicted','both','insufficient']
    answer:str|None
    evidence_ids:list[str]
    counterevidence_ids:list[str]
    confidence:float=Field(ge=0,le=1)
    explanation:str
    @model_validator(mode='after')
    def logical(self):
        if self.status=='both' and (not self.evidence_ids or not self.counterevidence_ids):
            raise ValueError('both needs evidence and counterevidence')
        if self.status=='insufficient' and self.answer is not None:
            raise ValueError('insufficient must abstain (answer=null)')
        if not self.explanation.strip():raise ValueError('explanation required')
        if self.status=='insufficient' and not any(word in self.explanation.lower() for word in ('неизвест','недостат','не хватает','нет дан','не установ','отсутств','не указан','не сообщ','не раскры','не содерж','нет информа','не определ')):
            raise ValueError('insufficient explanation must identify missing information')
        return self

def validate(raw,allowed,question_id):
    result=Answer.model_validate_json(raw)
    if result.question_id!=question_id:raise ValueError('question_id mismatch')
    if not set(result.evidence_ids+result.counterevidence_ids)<=set(allowed):
        raise ValueError('unknown or unavailable evidence_id')
    return result
