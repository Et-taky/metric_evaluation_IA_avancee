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
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay
X,y=load_digits(return_X_y=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,stratify=y,random_state=42)
print("Entraînement :",len(y_train),"; test :",len(y_test))
fig,axes=plt.subplots(1,8,figsize=(10,2))
for ax,x,label in zip(axes,X_test[:8],y_test[:8]):
    ax.imshow(x.reshape(8,8),cmap="gray"); ax.set_title(str(label)); ax.axis("off")
plt.tight_layout(); plt.savefig(ROOT/"results/digits.png",dpi=150); plt.show()

# %%
model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=42))
model.fit(X_train,y_train)
pred=model.predict(X_test)
baseline=DummyClassifier(strategy="most_frequent").fit(X_train,y_train)
print("Accuracy modèle :",round(accuracy_score(y_test,pred),4))
print("Accuracy référence naïve :",round(baseline.score(X_test,y_test),4))

# %%
report=classification_report(y_test,pred,output_dict=True,zero_division=0)
print(classification_report(y_test,pred,digits=3,zero_division=0))
summary={"accuracy":accuracy_score(y_test,pred),"macro_f1":report["macro avg"]["f1-score"],
         "weighted_f1":report["weighted avg"]["f1-score"],"n_test":len(y_test)}
(ROOT/"results/classification_scores.json").write_text(json.dumps(summary,indent=2))
pd.DataFrame(report).T.to_csv(ROOT/"results/classification_report.csv")

# %%
cm=confusion_matrix(y_test,pred,labels=range(10))
fig,ax=plt.subplots(figsize=(7,6))
ConfusionMatrixDisplay(cm,display_labels=range(10)).plot(ax=ax,cmap="Blues",colorbar=False)
ax.set_xlabel("Classe prédite");ax.set_ylabel("Vraie classe")
plt.tight_layout();plt.savefig(ROOT/"results/confusion.png",dpi=160);plt.show()
assert abs(cm.trace()/cm.sum()-summary["accuracy"])<1e-12
chiffre=8
vp=cm[chiffre,chiffre];fp=cm[:,chiffre].sum()-vp;fn=cm[chiffre,:].sum()-vp
print("Chiffre 8 : VP, FP, FN =",vp,fp,fn)
print("Précision :",vp/(vp+fp),"; rappel :",vp/(vp+fn))
