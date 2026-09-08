from pathlib import Path
import json
from streamlit.testing.v1 import AppTest
P=Path(__file__).parent
a=AppTest.from_file(str(P/'app.py')).run();assert not a.exception
comparison=a.dataframe[-1].value
assert list(comparison['Режим'])==['A','B','C']
assert list(comparison['Evidence F1'])==[0.0,1.0,1.0]
a.radio[0].set_value('B').run();assert not a.exception
assert any(x.value=='entailed' for x in a.metric)
assert any(x.value=='1.00' for x in a.metric)
(P/'results/ui_saved_results_check.json').write_text(json.dumps({'exceptions':0,'question':'Q01','modes':['A','B','C'],'displayed_evidence_f1':[0,1,1],'mode_B_status':'entailed','mode_B_confidence':'1.00'},indent=2),encoding='utf8')
print('Saved results, evidence F1, status and confidence displayed correctly.')
