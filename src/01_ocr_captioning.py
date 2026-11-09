# %%
# Dans Colab, décommenter si nécessaire :
# !pip -q install -r requirements.txt
# !apt-get -qq update && apt-get -qq install -y tesseract-ocr


# %%
from pathlib import Path
import sys
# Fonctionne depuis la racine du dépôt ou depuis notebooks/.
ROOT = Path.cwd() if (Path.cwd()/"src").exists() else Path.cwd().parent
assert (ROOT/"src").exists(), "Ouvrir le notebook dans le dépôt extrait"
sys.path.insert(0, str(ROOT))
(ROOT/"results").mkdir(exist_ok=True)
import json
import matplotlib.pyplot as plt
import pandas as pd


# %%
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import shutil, subprocess, tempfile
refs = ["TOTAL 1250 MAD", "FACTURE 2026", "CODE O0I1"]
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
if not Path(font_path).exists():
    # Alternative Windows : définir ici C:/Windows/Fonts/arial.ttf
    raise FileNotFoundError("Adapter font_path à une police TrueType installée")
font = ImageFont.truetype(font_path, 45)
images = []
for i, ref in enumerate(refs):
    image = Image.new("RGB", (600, 120), "white")
    ImageDraw.Draw(image).text((20,30), ref, font=font, fill="black")
    if i == 1:
        image = image.resize((120,24)).resize((600,120)).filter(ImageFilter.GaussianBlur(1.4))
    images.append(image)
    image.save(ROOT/f"data/ocr_{i}.png")
fig, axes = plt.subplots(3,1,figsize=(10,4))
for ax, image in zip(axes,images):
    ax.imshow(image); ax.axis("off")
plt.tight_layout(); plt.savefig(ROOT/"results/ocr_images.png",dpi=140); plt.show()


# %%
if not shutil.which("tesseract"):
    raise RuntimeError("Installer Tesseract puis relancer cette cellule")
preds = []
for i in range(len(images)):
    output = subprocess.run(["tesseract",str(ROOT/f"data/ocr_{i}.png"),"stdout",
                             "-l","eng","--psm","7"],capture_output=True,text=True,check=True)
    preds.append(" ".join(output.stdout.upper().split()))
refs = [" ".join(r.upper().split()) for r in refs]
print(pd.DataFrame({"Référence":refs,"Prédiction OCR":preds}).to_string(index=False))


# %%
from src.metrics import distance, error_rates
scores = error_rates(refs,preds)
print(pd.Series(scores).map(lambda x:f"{x:.2%}"))
(ROOT/"results/ocr_scores.json").write_text(json.dumps(scores,indent=2))
# Exemple vérifiable à la main : une substitution parmi 4 caractères.
assert distance("CHAT","CH4T") == 1
print("CHAT → CH4T : CER =", distance("CHAT","CH4T")/4,
      "; WER =", distance(["CHAT"],["CH4T"])/1)
# Des insertions peuvent rendre le taux supérieur à 100 %.
assert error_rates(["A"],["AAAA"])["CER"] == 3


# %%
rows=[]
for ref,pred in zip(refs,preds):
    rows.append({"Référence":ref,"OCR":pred,**error_rates([ref],[pred])})
table=pd.DataFrame(rows)
print(table.to_string(index=False))
table.to_csv(ROOT/"results/ocr_details.csv",index=False)


# %%
# Dans Colab si nécessaire :
# !pip -q install -r requirements.txt


# %%
from pathlib import Path
import sys
# Fonctionne depuis la racine du dépôt ou depuis notebooks/.
ROOT = Path.cwd() if (Path.cwd()/"src").exists() else Path.cwd().parent
assert (ROOT/"src").exists(), "Ouvrir le notebook dans le dépôt extrait"
sys.path.insert(0, str(ROOT))
(ROOT/"results").mkdir(exist_ok=True)
import json
import matplotlib.pyplot as plt
import pandas as pd


# %%
from sklearn.datasets import load_sample_images
from PIL import Image
sample=load_sample_images()
images=sample.images
references=[
 ["un bâtiment traditionnel près des arbres", "un édifice avec un jardin et des arbres"],
 ["une fleur orange devant un fond vert", "une fleur aux pétales orange dans la nature"]]
candidates={
 "Proches des références":["un bâtiment traditionnel près des arbres","une fleur orange devant un fond vert"],
 "Paraphrases":["une construction asiatique dans un jardin","des pétales orangés entourés de verdure"],
 "Erreurs de sens":["un bâtiment traditionnel près des voitures","une fleur bleue devant un fond vert"]}
fig,axes=plt.subplots(1,2,figsize=(10,4))
for i,(ax,image,refs) in enumerate(zip(axes,images,references)):
    ax.imshow(image);ax.axis("off");ax.set_title(refs[0],fontsize=10)
    Image.fromarray(image).save(ROOT/f"data/caption_{i}.jpg")
plt.tight_layout();plt.savefig(ROOT/"results/caption_images.png",dpi=140);plt.show()
print(json.dumps({"références":references,"candidats":candidates},ensure_ascii=False,indent=2))


# %%
from src.metrics import bleu_corpus,rouge_l
rows=[]
for name,preds in candidates.items():
    rows.append({"Candidats":name,"BLEU-1":bleu_corpus(references,preds,n=1),
                 "BLEU-4":bleu_corpus(references,preds,n=4),
                 "ROUGE-L F1":sum(max(rouge_l(r,p) for r in refs)
                                     for refs,p in zip(references,preds))/len(preds)})
table=pd.DataFrame(rows).set_index("Candidats")
print(table.round(3).to_string())
table.to_csv(ROOT/"results/caption_scores.csv")
assert abs(table.iloc[0]["BLEU-4"]-1)<1e-12
assert rouge_l("un chat noir","un chat noir")==1
assert bleu_corpus([["un chat noir"]],["un chien blanc"],n=4)==0


# %%
ax=table.plot.bar(figsize=(9,4),ylim=(0,1),rot=12)
ax.set_ylabel("Score entre 0 et 1");ax.set_xlabel("")
plt.tight_layout();plt.savefig(ROOT/"results/caption_scores.png",dpi=150);plt.show()
for i,refs in enumerate(references):
    print("Image",i+1,"| référence :",refs[0])
    print("Erreur contrôlée :",candidates["Erreurs de sens"][i])
