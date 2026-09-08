from pathlib import Path
import os,json,re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from razdel import tokenize, sentenize
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

HERE=Path(__file__).resolve().parent

def run():
    out=HERE/'results';out.mkdir(exist_ok=True)
    df=pd.read_csv(HERE/'data/texts.csv')
    audit={'n':len(df),'missing':df.isna().sum().to_dict(),'duplicate_texts':int(df.text.duplicated().sum()),'class_counts':df.label.value_counts().to_dict()}
    features=pd.DataFrame({'id':df.id,'characters':df.text.str.len(),'tokens':df.text.map(lambda t:len(list(tokenize(t)))),'sentences':df.text.map(lambda t:len(list(sentenize(t)))),'exclamations':df.text.str.count('!'),'questions':df.text.str.count(r'\?'),'uppercase_ratio':df.text.map(lambda t:sum(c.isupper() for c in t)/max(1,sum(c.isalpha() for c in t)))})
    features.to_csv(out/'surface_features.csv',index=False); audit['lengths']=features[['characters','tokens']].describe().to_dict()
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    transformations=[
      ('random_00470','reorder','Обожаю её))) Спокойная хорошая песенка)))',False,'Переставлены предложения, положительная оценка сохранена.'),
      ('random_00470','paraphrase','Мне очень нравится эта приятная спокойная песня.',False,'Выражено то же восхищение спокойной песней другими словами.'),
      ('random_00470','meaning_change','Спокойная хорошая песенка))) Не обожаю её)))',True,'Отрицание меняет отношение автора: восхищение больше не утверждается.'),
      ('random_20278','reorder','У тебя в последнее время одни глупые репосты.',False,'Перемещение обстоятельства не меняет отрицательную оценку репостов.'),
      ('random_20278','paraphrase','Последние перепубликации на твоей странице все бестолковые.',False,'Сохраняются адресат, временная область и отрицательное отношение.'),
      ('random_20278','meaning_change','В последнее время у тебя не одни глупые репосты.',True,'Отрицается всеобщность: теперь допускаются и неглупые публикации.'),
      ('random_05842','reorder','Роднуля, когда выезжаешь?))))))',False,'Обращение перенесено, вопрос о времени выезда сохранён.'),
      ('random_05842','paraphrase','Во сколько отправляешься, дорогой человек?',False,'Переформулирован вопрос о времени отправления; оттенок обращения ослаблен.'),
      ('random_05842','meaning_change','Когда приезжаешь, роднуля?))))))',True,'Вместо времени выезда спрашивается время прибытия.'),
    ]
    rows=[]
    for ident,kind,text,changed,reason in transformations:
        source=df.set_index('id').loc[ident]
        rows.append(dict(id=ident,label=source.label,original=source.text,transformation=kind,transformed=text,meaning_changed=changed,judgment=reason,reviewer='Codex analytical judgment, not human participant'))
    pairs=pd.DataFrame(rows); pairs.to_csv(HERE/'data/transformations.csv',index=False)
    tok=lambda text:[t.text.lower() for t in tokenize(text) if any(c.isalnum() for c in t.text)]
    bow=CountVectorizer(tokenizer=tok,token_pattern=None,lowercase=False)
    tfidf=TfidfVectorizer(tokenizer=tok,token_pattern=None,lowercase=False)
    for name,vectorizer in [('bow',bow),('tfidf',tfidf)]:
        matrix=vectorizer.fit_transform(df.text)
        np.savez_compressed(out/f'{name}_matrix.npz',data=matrix.data,indices=matrix.indices,indptr=matrix.indptr,shape=matrix.shape)
        (out/f'{name}_vocabulary.json').write_text(json.dumps(vectorizer.get_feature_names_out().tolist(),ensure_ascii=False),encoding='utf-8')
        original=vectorizer.transform(pairs.original); altered=vectorizer.transform(pairs.transformed)
        pairs[name]=[float(cosine_similarity(original[i],altered[i])[0,0]) for i in range(9)]
        pairs[f'{name}_oov_ratio']=pairs.transformed.map(lambda s:sum(t not in vectorizer.vocabulary_ for t in tok(s))/max(1,len(tok(s))))
    local=Path(os.getenv('AOT_EMBEDDING_PATH','E:/AI/models/embeddings/multilingual-e5-small'))
    source=str(local) if local.exists() else 'intfloat/multilingual-e5-small'
    model=SentenceTransformer(source,device='cpu')
    embeddings=model.encode(['query: '+t for t in df.text],normalize_embeddings=True,batch_size=16,show_progress_bar=False)
    np.save(out/'embeddings.npy',embeddings)
    em=model.encode(['query: '+t for t in pairs.original]+['query: '+t for t in pairs.transformed],normalize_embeddings=True,batch_size=16,show_progress_bar=False)
    pairs['embeddings']=(em[:9]*em[9:]).sum(axis=1)
    pairs.to_csv(out/'similarities.csv',index=False)
    (out/'embedding_config.json').write_text(json.dumps({'model':'intfloat/multilingual-e5-small','device':'cpu','prefix':'query: for both sides (symmetric similarity)','normalize_embeddings':True,'max_seq_length':model.max_seq_length,'dimensions':int(embeddings.shape[1])},indent=2))
    sns.set_theme(style='whitegrid',font='DejaVu Sans')
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    axes[0].hist(features.characters,bins=20,color='#436B93');axes[0].set(xlabel='Символы',ylabel='Число текстов',title='Длина текста в символах')
    axes[1].hist(features.tokens,bins=20,color='#58836B');axes[1].set(xlabel='Токены razdel',ylabel='Число текстов',title='Длина текста в токенах')
    fig.tight_layout();fig.savefig(out/'length_distribution.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    labels=[f"{r.label}: {dict(reorder='перестановка',paraphrase='парафраз',meaning_change='изменение смысла')[r.transformation]}" for r in pairs.itertuples()]
    sns.heatmap(pairs[['bow','tfidf','embeddings']],annot=True,fmt='.3f',vmin=0,vmax=1,cmap='YlGnBu',yticklabels=labels,ax=ax)
    ax.set(xlabel='Представление',ylabel='');fig.tight_layout();fig.savefig(out/'similarity_heatmap.png',dpi=180);plt.close(fig)
    print(pairs[['label','transformation','meaning_changed','bow','tfidf','embeddings']].to_string(index=False))
    return pairs
if __name__=='__main__':run()
