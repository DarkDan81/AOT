from pathlib import Path
p=Path(__file__).parent/'run_experiment.py';s=p.read_text(encoding='utf-8-sig');s=s.replace("row.update(valid=False,error=type(e).__name__+': '+str(e),elapsed_seconds=time.perf_counter()-start)","row.update(valid=False,error=type(e).__name__+': '+str(e),elapsed_seconds=time.perf_counter()-start)\n    if hasattr(e,'response'):row['transport']=e.response")
p.write_text(s,encoding='utf8')
