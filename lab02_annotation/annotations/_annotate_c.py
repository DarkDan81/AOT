import csv
from pathlib import Path
ones={1,3,5,8,10,11,13,15,19,21,24,26,29,31,32,34,37,38,41,43,45,47,52,53,56,59,63,66,69,70,72,75,78,82,84,87,88}
uncertain={17,22,27,33,49,57,60,65,73,85}
rows=list(csv.DictReader(Path('lab02_annotation/data/texts.csv').open(encoding='utf-8-sig')))
with Path('lab02_annotation/annotations/initial_c.csv').open('w',encoding='utf8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['id','label','reason']);w.writeheader()
 for i,row in enumerate(rows,1):
  label='1' if i in ones else '?' if i in uncertain else '0'
  reason='Явное ускорение или непосредственно опасная ситуация.' if label=='1' else 'Проблема или срок есть, но необходимость немедленного действия не вполне определена.' if label=='?' else 'Нет актуального требования немедленного действия; обычный срок, фон, цитата либо отрицание.'
  w.writerow(dict(id=row['id'],label=label,reason=reason))
