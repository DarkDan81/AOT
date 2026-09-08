from pathlib import Path
p=Path(__file__).parent/'src/schemas.py';s=p.read_text(encoding='utf8');s=s.replace("if not self.explanation.strip():raise ValueError('explanation required')", "if not self.explanation.strip():raise ValueError('explanation required')\n        if self.status=='insufficient' and not any(word in self.explanation.lower() for word in ('неизвест','недостат','не хватает','нет дан','не установ','отсутств','не указан','не сообщ','не раскры','не содерж','нет информа','не определ')):\n            raise ValueError('insufficient explanation must identify missing information')")
p.write_text(s,encoding='utf8')
p=Path(__file__).parent/'tests/test_system.py';s=p.read_text(encoding='utf-8-sig');s+='''\n\ndef test_insufficient_fabricated_explanation():
 with pytest.raises(ValueError):validate(raw(status='insufficient',answer=None,explanation='Илья удалил файл из мести.'),['C08'],'Q01')

def test_raw_invalid_response_retained():
 from src.llm import ask,ResponseValidationError
 response={'content':'not json','usage':{'prompt_tokens':10}}
 with patch('src.llm.complete',return_value=response):
  with pytest.raises(ResponseValidationError) as e:ask(QUESTIONS[0],'context',['C08'])
 assert e.value.response==response
''';p.write_text(s,encoding='utf8')
