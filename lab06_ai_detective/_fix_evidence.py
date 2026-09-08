from pathlib import Path
p=Path(__file__).parent/'build_variants.py';s=p.read_text(encoding='utf8');needle="reason='Добавлено одно несовместимое утверждение."
insert="""if q['question_id']=='Q03':vq['gold_evidence_ids']=['C37','C07']
   if q['question_id']=='Q05':vq['gold_evidence_ids']=['C37','C07']
   if q['question_id']=='Q04':vq['gold_counterevidence_ids']=['C37','C02']
   if q['question_id']=='Q06':vq['gold_counterevidence_ids']=['C37','C12']
   if q['question_id']=='Q07':vq['gold_counterevidence_ids']=['C37','C10','C04']
   """
s=s.replace(needle,insert+needle);p.write_text(s,encoding='utf8')
