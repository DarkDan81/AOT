from pathlib import Path
import pymupdf as fitz
from PIL import Image,ImageDraw
for p in Path('.').glob('lab*/report.pdf'):
 d=fitz.open(p);out=Path('tmp/qa')/p.parent.name;out.mkdir(parents=True,exist_ok=True)
 for i,page in enumerate(d):page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(out/f'page-{i+1}.png')
 for start in range(0,len(d),8):
  canvas=Image.new('RGB',(1200,1800),'#dddddd');draw=ImageDraw.Draw(canvas)
  for j in range(start,min(start+8,len(d))):
   im=Image.open(out/f'page-{j+1}.png');im.thumbnail((580,420));x=((j-start)%2)*600+10;y=((j-start)//2)*450+20;canvas.paste(im,(x,y));draw.text((x,y-15),str(j+1),fill='black')
  canvas.save(out/f'contact-{start//8+1}.png')
 print(p,len(d))
