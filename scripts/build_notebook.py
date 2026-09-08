from pathlib import Path
import ast,textwrap
import nbformat
from nbclient import NotebookClient
p=Path('lab01_representations')
src=(p/'run.py').read_text(encoding='utf-8')
imports=src.split('HERE=')[0]+'\nHERE=Path.cwd() if Path.cwd().name=="lab01_representations" else Path.cwd()/"lab01_representations"\n'
body=src.split('def run():\n',1)[1].split("if __name__",1)[0]
body=textwrap.dedent(body).replace('return pairs','display(pairs)')
markers=['transformations=[','rows=[]','tok=lambda','local=Path','sns.set_theme','fig,ax=plt.subplots']
chunks=[];cur=0
for marker in markers:
    idx=body.index(marker); chunks.append(body[cur:idx]);cur=idx
chunks.append(body[cur:])
heads=['1. Данные, пропуски, дубликаты и поверхностные признаки','2. Девять контролируемых преобразований','3. Сохранение исходных и изменённых текстов','4. Bag-of-words и TF–IDF','5. Sentence embeddings (CPU)','6. Распределение длин','7. Визуализация и сравнения']
cells=[nbformat.v4.new_markdown_cell('# Лабораторная работа №1\nДущенко Даниил Александрович, К3341. Все ячейки выполняются сверху вниз. Эмбеддинги работают на CPU; локальный путь можно задать AOT_EMBEDDING_PATH.'),nbformat.v4.new_code_cell(imports)]
for h,c in zip(heads,chunks):cells.extend([nbformat.v4.new_markdown_cell('## '+h),nbformat.v4.new_code_cell(c)])
cells.append(nbformat.v4.new_code_cell("from IPython.display import Image, display\ndisplay(Image(filename=str(HERE/'results/length_distribution.png')))\ndisplay(Image(filename=str(HERE/'results/similarity_heatmap.png')))"))
nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python (AOT)','language':'python','name':'aot'},'language_info':{'name':'python'}})
NotebookClient(nb,timeout=600,kernel_name='aot',resources={'metadata':{'path':str(p.resolve())}}).execute()
nbformat.write(nb,p/'lab01.ipynb')
print('Notebook executed,',len(cells),'cells')
