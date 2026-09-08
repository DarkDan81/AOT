import json
from pathlib import Path
from streamlit.testing.v1 import AppTest
P=Path(__file__).parent
qs=[json.loads(x) for x in (P/'data/questions.jsonl').read_text(encoding='utf8').splitlines()]
a=AppTest.from_file(str(P/'app.py')).run();checks=[]
for qi in (0,2,8,10):
 a.selectbox[0].set_value(qs[qi]).run()
 for mutation in ('baseline','permutation','deletion','contradiction'):
  a.selectbox[1].set_value(mutation).run()
  for mode in ('A','B','C'):
   a.radio[0].set_value(mode).run()
   assert not a.exception,(qi,mutation,mode,a.exception)
   checks.append([qs[qi]['question_id'],mutation,mode])
(P/'results/ui_checks.json').write_text(json.dumps({'checked_states':checks,'count':len(checks),'exceptions':0},indent=2),encoding='utf8')
print('UI states passed:',len(checks))
