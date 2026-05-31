"""
Stratified 5-fold cross-validation untuk model klasik (fitur TF-IDF).
Menguji ketahanan/stabilitas hasil pada 4 algoritma.

Cara pakai:
    python src/run_crossval.py --data data/Indonesian_Sentiment_Twitter_Dataset_Labeled.csv
"""
import argparse, json, os, re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

SEED = 42
np.random.seed(SEED)


def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"http\S+|www\S+", " ", t)
    t = re.sub(r"@\w+|#\w+", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"(.)\1{2,}", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.data, sep="\t").drop_duplicates().dropna().reset_index(drop=True)

    stemmer = StemmerFactory().create_stemmer()
    stopwords = set(StopWordRemoverFactory().get_stop_words())
    cache = {}

    def preprocess(text):
        out = []
        for w in clean_text(text).split():
            if w in stopwords or len(w) <= 1:
                continue
            if w not in cache:
                cache[w] = stemmer.stem(w)
            out.append(cache[w])
        return " ".join(out)

    print("Preprocessing...")
    df["proc"] = df["Tweet"].apply(preprocess)
    df = df[df["proc"].str.strip() != ""].reset_index(drop=True)
    X, y = df["proc"].values, df["sentimen"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "SVM": SVC(kernel="linear", C=1.0, random_state=SEED),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    }
    cv_res = {}
    for name, clf in models.items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)),
            ("clf", clf)])
        scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
        cv_res[name] = {"mean": round(scores.mean()*100, 2),
                        "std": round(scores.std()*100, 2),
                        "folds": [round(s*100, 2) for s in scores]}
        print(f"{name:22s} {scores.mean()*100:.2f} +/- {scores.std()*100:.2f}", flush=True)

    json.dump(cv_res, open(os.path.join(args.outdir, "cv_results.json"), "w"), indent=2)
    print(f"\nHasil disimpan ke {args.outdir}/cv_results.json")


if __name__ == "__main__":
    main()
