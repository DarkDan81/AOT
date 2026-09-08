from pathlib import Path
from common.llm import complete
from .schemas import Answer,validate
PROMPT=(Path(__file__).parents[1]/'prompts/system.txt').read_text(encoding='utf-8-sig')
class ResponseValidationError(ValueError):
    def __init__(self,error,response):
        super().__init__(str(error));self.response=response

def ask(question,context,allowed):
    response=complete([{'role':'system','content':PROMPT},{'role':'user','content':f"ВОПРОС {question['question_id']}: {question['question']}\nКОНТЕКСТ:\n{context}"}],schema=Answer.model_json_schema(),max_tokens=1500)
    try:validated=validate(response['content'],allowed,question['question_id'])
    except ValueError as error:raise ResponseValidationError(error,response) from error
    return validated.model_dump(),response
