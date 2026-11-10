# Évaluation de trois sujets de vision par ordinateur

Master Data & IA. Organisation actuelle :

1. **Classification d’images**.
2. **OCR**.
3. **Génération d’images**.

## Exécution locale

Python 3.10 ou supérieur, CPU suffisant. Depuis la racine du dossier extrait :

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux/macOS : source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install jupyterlab
jupyter lab
```

L’OCR nécessite le programme **Tesseract** et les données `eng`. Sous Debian/Ubuntu :

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

Sous Windows, suivre [l’installation Tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html), ajouter le programme au `PATH`, puis vérifier `tesseract --version`. Dans le notebook fusionné, adapter la variable `font_path`, par exemple `C:/Windows/Fonts/arial.ttf`. Le changement de police peut modifier les sorties OCR.

Les versions Python se lancent depuis la racine :

```bash
python src/02_classification_images.py
python src/01_ocr_captioning.py
python src/03_image_generation.py
```

## Exécution dans Google Colab

1. Ouvrir le notebook souhaité dans Colab.
2. Importer aussi `Projet_Evaluation_CV_Complet.zip` dans le panneau Fichiers.
3. Ajouter cette cellule au début :

```python
!unzip -q -o /content/Projet_Evaluation_CV_Complet.zip -d /content
%cd /content/projet_evaluation_cv
!pip -q install -r requirements.txt
# Uniquement pour le notebook OCR et captioning :
!apt-get -qq update && apt-get -qq install -y tesseract-ocr fonts-dejavu-core
```

4. Exécuter les cellules de haut en bas. Aucun GPU ni téléchargement de modèle lourd n’est requis.
5. ou bien juste upload les notebook et exécute tout simplement le dépot Github fournit tout

## Classification

- Vraies images digits, 1 797 images 8 × 8, dix classes.
- Découpage stratifié : 1 347 exemples d’entraînement, 450 de test, graine 42.
- Régression logistique et standardisation dans une pipeline, apprentissage uniquement sur l’entraînement.
- Accuracy, précision, rappel et F1 par classe, macro-F1, F1 pondéré, matrice de confusion et référence naïve.
- Résultats observés : accuracy 97,78 %, macro-F1 0,9776.
- Les grilles et matrices comptées à la main dans les slides sont des exemples pédagogiques distincts des résultats du notebook.

## OCR

### OCR

- Tesseract reconnaît trois lignes dont la vérité terrain est connue à l’avance.
- CER et WER utilisent une distance d’édition à coûts unitaires.
- Normalisation identique : casse ignorée et espaces répétés réduits. Le CER inclut les espaces restants ; le WER découpe sur les espaces.
- Agrégation : total des éditions divisé par le total des unités de référence.
- Scores observés : CER 2,86 %, WER 14,29 %, deux lignes exactes sur trois. Ils peuvent varier selon Tesseract et la police.

## Génération d’images

### Modèle réellement exécuté

Le notebook apprend une PCA à 20 composantes sur l’entraînement, puis un mélange de deux gaussiennes par classe. Il échantillonne 45 points par chiffre et reconstitue 450 nouvelles images 8 × 8. Un contre-exemple répète une seule vraie image 450 fois.

### Mesures calculées

| Ensemble         | Fréchet-PCA | Classes reconnues | Doublons |
| ---------------- | -----------: | ----------------: | -------: |
| Générateur     |       0,1283 |                10 |      0 % |
| Image répétée |       6,5120 |                 1 |  99,78 % |

Le nombre de classes provient d’un classifieur de chiffres distinct du générateur, appris sur l’entraînement. L’accord avec la classe demandée vaut ici 98 % pour les images générées ; cet indicateur n’est pas une preuve de qualité visuelle. Les doublons sont comptés après arrondi à six décimales.

**Fréchet-PCA n’est pas le FID standard.** Le notebook utilise la formule de distance de Fréchet sur des caractéristiques PCA. Le FID standard emploie des caractéristiques Inception préentraînées. Ces nombres ne doivent donc pas être appelés FID Inception ni comparés à des valeurs de publications.

### Métriques expliquées

- FID : rapprochement des statistiques réelles et générées dans l’espace Inception, plus faible généralement souhaité.
- Inception Score : confiance individuelle et diversité des classes, plus élevé généralement souhaité ; ne compare pas directement aux images réelles.
- Diversité et inspection visuelle : plusieurs classes, variantes et défauts éventuels.
- CLIPScore : complément pour l’accord image-texte dans la génération guidée par une consigne.

Le FID standard, l’Inception Score et CLIPScore sont **expliqués mais non calculés**. Le choix pédagogique permet d’exécuter tout le notebook sur CPU sans télécharger de réseau. Les familles GAN, VAE et diffusion sont citées comme autres modèles génératifs.
