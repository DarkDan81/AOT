from pathlib import Path
p=Path(__file__).parent/'build_variants.py';s=p.read_text(encoding='utf-8-sig')
s=s.replace("elif q['gold_status']=='both':", "if q['gold_status']=='contradicted':\n    vq['gold_evidence_ids']=['C37'];vq['gold_counterevidence_ids']=q['gold_evidence_ids']\n   elif q['gold_status']=='both':")
p.write_text(s,encoding='utf8')
p=Path(__file__).parent/'prompts/system.txt';s=p.read_text(encoding='utf8');s=s.replace('в counterevidence_ids — противоположной позиции при both.','в counterevidence_ids — противоположной позиции при both. При both evidence_ids подтверждают утвердительную пропозицию вопроса, counterevidence_ids опровергают её. При contradicted evidence_ids содержат основания отрицательного ответа (это соглашение оценки).')
p.write_text(s,encoding='utf8')
