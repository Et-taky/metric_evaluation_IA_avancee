# %%
# Si nécessaire dans Colab :
# !pip -q install -r requirements.txt
from pathlib import Path
import sys, json
ROOT=Path.cwd() if (Path.cwd()/"src").exists() else Path.cwd().parent
assert (ROOT/"src").exists(), "Ouvrir le notebook dans le projet extrait"
sys.path.insert(0,str(ROOT))
(ROOT/"results").mkdir(exist_ok=True)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# %%
X,y=load_digits(return_X_y=True)
X=X/16.0
X_train,X_real,y_train,y_real=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
pca=PCA(n_components=20,random_state=42).fit(X_train)
Z_train=pca.transform(X_train)
print("Entraînement :",len(X_train),"; référence réelle :",len(X_real))
print("Variance conservée par la PCA :",round(pca.explained_variance_ratio_.sum(),3))


# %%
generated=[];requested=[]
for digit in range(10):
    generator=GaussianMixture(n_components=2,covariance_type="full",reg_covar=.02,random_state=42+digit)
    generator.fit(Z_train[y_train==digit])
    latent,_=generator.sample(45)
    generated.append(np.clip(pca.inverse_transform(latent),0,1))
    requested.extend([digit]*45)
X_gen=np.vstack(generated);requested=np.array(requested)
# Contre-exemple volontaire : répéter une seule vraie image 450 fois.
one=X_train[y_train==8][0]
X_repeat=np.repeat(one[None,:],len(X_gen),axis=0)
np.savez(ROOT/"data/generation_chiffres.npz",real=X_real,generated=X_gen,repeated=X_repeat,requested=requested)
fig,axes=plt.subplots(3,10,figsize=(13,4.8))
for col in range(10):
    real=X_real[y_real==col][0]
    for row,image in enumerate([real,X_gen[col*45],X_repeat[col]]):
        axes[row,col].imshow(image.reshape(8,8),cmap="gray",vmin=0,vmax=1)
        axes[row,col].axis("off")
    axes[0,col].set_title(str(col),fontsize=14)
for ax,label in zip(axes[:,0],["Réel","Généré","Répété"]):
    ax.text(-.25,.5,label,transform=ax.transAxes,ha="right",va="center",rotation=90,fontsize=12)
plt.tight_layout();plt.savefig(ROOT/"results/generation_exemples.png",dpi=180,bbox_inches="tight");plt.show()
print("450 images générées, avec 45 demandes de chaque chiffre.")


# %%
def frechet_features(a,b):
    mu_a,mu_b=a.mean(axis=0),b.mean(axis=0)
    cov_a,cov_b=np.cov(a,rowvar=False),np.cov(b,rowvar=False)
    eigenvalues,vectors=np.linalg.eigh(cov_a)
    sqrt_a=(vectors*np.sqrt(np.clip(eigenvalues,0,None)))@vectors.T
    product=sqrt_a@cov_b@sqrt_a
    product=(product+product.T)/2
    trace_root=np.sqrt(np.clip(np.linalg.eigvalsh(product),0,None)).sum()
    value=np.sum((mu_a-mu_b)**2)+np.trace(cov_a)+np.trace(cov_b)-2*trace_root
    return float(max(0,value))
Z_real=pca.transform(X_real)
frechet_gen=frechet_features(Z_real,pca.transform(X_gen))
frechet_repeat=frechet_features(Z_real,pca.transform(X_repeat))
assert frechet_features(Z_real,Z_real)<1e-8
print("Fréchet-PCA, générateur :",round(frechet_gen,4))
print("Fréchet-PCA, répétition :",round(frechet_repeat,4))


# %%
evaluator=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=42)).fit(X_train,y_train)
print("Accuracy du classifieur sur les vraies images de test :",round(evaluator.score(X_real,y_real),4))
pred_gen=evaluator.predict(X_gen);pred_repeat=evaluator.predict(X_repeat)
def diversity(images,pred):
    unique=len(np.unique(np.round(images,6),axis=0))
    return len(np.unique(pred)),1-unique/len(images)
classes_gen,duplicates_gen=diversity(X_gen,pred_gen)
classes_repeat,duplicates_repeat=diversity(X_repeat,pred_repeat)
summary=pd.DataFrame([
 {"Ensemble":"Générateur","Fréchet-PCA":frechet_gen,"Classes reconnues":classes_gen,"Doublons":duplicates_gen,"Accord avec demande":float(np.mean(pred_gen==requested))},
 {"Ensemble":"Image répétée","Fréchet-PCA":frechet_repeat,"Classes reconnues":classes_repeat,"Doublons":duplicates_repeat,"Accord avec demande":float(np.mean(pred_repeat==requested))}]).set_index("Ensemble")
print(summary.round(4).to_string())
summary.to_csv(ROOT/"results/generation_scores.csv")
(ROOT/"results/generation_scores.json").write_text(summary.to_json(orient="index",indent=2,force_ascii=False))
fig,axes=plt.subplots(1,2,figsize=(10,3.5))
axes[0].hist(pred_gen,bins=np.arange(11)-.5,rwidth=.8,color="#216F5B")
axes[1].hist(pred_repeat,bins=np.arange(11)-.5,rwidth=.8,color="#A74D2C")
for ax,title in zip(axes,["Générateur : classes reconnues","Répétition : une seule classe"]):
    ax.set_xticks(range(10));ax.set_title(title);ax.set_xlabel("Chiffre reconnu");ax.set_ylabel("Nombre d’images")
plt.tight_layout();plt.savefig(ROOT/"results/generation_diversite.png",dpi=170);plt.show()
assert len(X_gen)==len(X_real)==len(X_repeat)==450
assert classes_repeat==1 and duplicates_repeat>.99
