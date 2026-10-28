from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import json,hashlib
B=Path(__file__).resolve().parent
files=[]
for glob in ['*.py','*.md','*.docx','*.json','*.png']:
    files += [p for p in B.glob(glob) if p.name not in ('report_montage.png',)]
S=B/'q16_27_reference'
if not S.exists():
    from shutil import copytree
    S0=Path('/mnt/data/QMATH_16_28/q16_27_reference')
    copytree(S0,S,ignore=lambda d,names:{n for n in names if n=='__pycache__'})
files += [p for p in S.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
unique={str(p.relative_to(B)):p for p in files}
manifest={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in sorted(unique.items()) if not n.endswith('.pyc')}
(B/'SHA256_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
zip_path=B/'Q-MATH-16.29_同磁态材料双重证书_原始M0可复算研究包.zip'
with ZipFile(zip_path,'w',ZIP_DEFLATED) as z:
    for n,p in sorted(unique.items()):z.write(p,arcname=n)
    z.write(B/'SHA256_MANIFEST.json',arcname='SHA256_MANIFEST.json')
print(zip_path,zip_path.stat().st_size,'entries',len(unique)+1)
