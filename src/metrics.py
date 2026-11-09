"""Métriques pédagogiques : édition, BLEU corpus et ROUGE-L."""
import re, math
from collections import Counter

def tokens(text):
    # Même convention pour candidats et références ; accents conservés.
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

def distance(a, b):
    """Distance de Levenshtein : substitution, suppression et insertion coûtent 1."""
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j-1] + (x != y)))
        row = new
    return row[-1]

def error_rates(refs, preds):
    """Micro agrégation : total des éditions / total des unités de référence."""
    if len(refs) != len(preds) or not refs:
        raise ValueError("Listes non vides et de même taille requises")
    # Espaces inclus pour le CER ; WER découpé sur espaces.
    chars = sum(len(r) for r in refs)
    words = sum(len(r.split()) for r in refs)
    if not chars or not words:
        raise ValueError("Références vides non prises en charge")
    return {"CER": sum(distance(r,p) for r,p in zip(refs,preds))/chars,
            "WER": sum(distance(r.split(),p.split()) for r,p in zip(refs,preds))/words,
            "Exactitude": sum(r == p for r,p in zip(refs,preds))/len(refs)}

def ngrams(seq, n):
    return Counter(tuple(seq[i:i+n]) for i in range(len(seq)-n+1))

def bleu_corpus(references, predictions, n=4):
    """BLEU-N non lissé, poids uniformes, précision écrêtée et pénalité de brièveté.
    references : liste de listes de chaînes (plusieurs références par image).
    """
    clipped, totals = [0]*n, [0]*n
    c = r = 0
    for refs, pred in zip(references, predictions):
        hyp = tokens(pred); refs = [tokens(x) for x in refs]
        c += len(hyp)
        r += min((len(x) for x in refs), key=lambda z:(abs(z-len(hyp)),z))
        for k in range(1,n+1):
            counts=ngrams(hyp,k); maximum=Counter()
            for ref in refs:
                maximum |= ngrams(ref,k)
            clipped[k-1] += sum(min(v,maximum[g]) for g,v in counts.items())
            totals[k-1] += sum(counts.values())
    if c == 0 or any(v == 0 for v in clipped): return 0.0
    p=[a/b for a,b in zip(clipped,totals)]
    return math.exp(min(0,1-r/c) + sum(math.log(x) for x in p)/n)

def rouge_l(ref, pred):
    """F1 de la plus longue sous-séquence commune, beta=1, sans stemming."""
    a,b=tokens(ref),tokens(pred); row=[0]*(len(b)+1)
    for x in a:
        new=[0]
        for j,y in enumerate(b,1):
            new.append(row[j-1]+1 if x==y else max(row[j],new[-1]))
        row=new
    return 2*row[-1]/(len(a)+len(b)) if a or b else 1.0
