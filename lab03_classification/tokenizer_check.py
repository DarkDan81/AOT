"""Use native tokenizer of the ALREADY loaded model; never load/reload a model."""
import json
import time
import lmstudio
import pandas as pd
from .run import HERE
from common.llm import MODEL

def main():
    texts=pd.read_csv(HERE/'data/train.csv').head(10)
    with lmstudio.Client(api_host='localhost:1234') as client:
        loaded=client.llm.list_loaded()
        matching=[m for m in loaded if m.identifier==MODEL]
        if len(matching)!=1:raise RuntimeError(f'Required model must already be loaded: {MODEL}; found {[m.identifier for m in loaded]}')
        model=matching[0]
        results=[]
        for row in texts.to_dict('records'):
            started=time.perf_counter();ids=list(model.tokenize(row['text']));elapsed=time.perf_counter()-started
            results.append({'id':row['id'],'text':row['text'],'token_ids':ids,'n_tokens':len(ids),'tokenize_seconds':elapsed})
    (HERE/'results/native_tokenizer.json').write_text(json.dumps({'model':MODEL,'sdk':'lmstudio 1.5.0','method':'already_loaded_model.tokenize','includes_chat_template':False,'note':'Raw text tokenizer demonstration. Main experiment uses API usage including full chat template, system prompt and examples; counts intentionally differ.','records':results},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Native tokenizer verified on',len(results),'train texts')

if __name__=='__main__':main()
