"""
Analisis Sentimen Bahasa Indonesia - Model Klasik
Perbandingan 4 algoritma (Logistic Regression, SVM, Naive Bayes, Random Forest)
dengan 2 representasi fitur (TF-IDF, Word2Vec) = 8 kombinasi model.

Dataset: Indonesian Sentiment Twitter Dataset (Ferdiana dkk., 2019)
Cara pakai:
    python src/run_classic.py --data path/ke/Indonesian_Sentiment_Twitter_Dataset_Labeled.csv
"""
import argparse, json, os, re, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)
from gensim.models import Word2Vec
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

SEED = 42
np.random.seed(SEED)


def clean_text(t):
    """Case folding + pembersihan derau teks tweet."""
    t = str(t).lower()
    t = re.sub(r"http\S+|www\S+", " ", t)          # hapus URL
    t = re.sub(r"@\w+|#\w+", " ", t)                # hapus mention & tagar
    t = re.sub(r"[^a-z\s]", " ", t)                 # sisakan huruf
    t = re.sub(r"(.)\1{2,}", r"\1", t)              # normalisasi huruf berulang
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_preprocessor():
    """Bangun fungsi preprocessing dengan stemming ter-cache (efisien)."""
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

    return preprocess


def doc_vector(text, w2v, dim):
    """Rata-rata vektor kata (Word2Vec) untuk satu dokumen."""
    toks = [w for w in text.split() if w in w2v.wv]
    if not toks:
        return np.zeros(dim)
    return np.mean([w2v.wv[w] for w in toks], axis=0)


def evaluate(name, y_true, y_pred, results, cms):
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    results[name] = {"accuracy": round(acc * 100, 2),
                     "precision": round(p * 100, 2),
                     "recall": round(r * 100, 2),
                     "f1": round(f * 100, 2)}
    cms[name] = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1]).tolist()
    print(f"{name:30s} acc={acc*100:5.2f} prec={p*100:5.2f} "
          f"rec={r*100:5.2f} f1={f*100:5.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path CSV berlabel (tab-separated)")
    ap.add_argument("--outdir", default="results", help="Folder output")
    ap.add_argument("--w2v_dim", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 1. Muat & bersihkan
    df = pd.read_csv(args.data, sep="\t").drop_duplicates().dropna().reset_index(drop=True)
    print("Jumlah data awal (setelah dedup):", len(df))

    # 2. Preprocessing
    print("Preprocessing (stemming ter-cache)...")
    t0 = time.time()
    pre = build_preprocessor()
    df["proc"] = df["Tweet"].apply(pre)
    df = df[df["proc"].str.strip() != ""].reset_index(drop=True)
    print(f"  selesai dalam {time.time()-t0:.0f} dtk; data valid: {len(df)}")
    print("Distribusi kelas:", df["sentimen"].value_counts().to_dict())

    # 3. Split
    X, y = df["proc"].values, df["sentimen"].values
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y)
    print(f"Train: {len(Xtr)}  Test: {len(Xte)}")

    results, cms = {}, {}

    # 4a. TF-IDF
    print("\n=== TF-IDF ===")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    Xtr_t = tfidf.fit_transform(Xtr)
    Xte_t = tfidf.transform(Xte)
    evaluate("Logistic Regression + TF-IDF", yte,
             LogisticRegression(max_iter=1000, random_state=SEED).fit(Xtr_t, ytr).predict(Xte_t),
             results, cms)
    evaluate("SVM + TF-IDF", yte,
             SVC(kernel="linear", C=1.0, random_state=SEED).fit(Xtr_t, ytr).predict(Xte_t),
             results, cms)
    evaluate("Naive Bayes + TF-IDF", yte,
             MultinomialNB().fit(Xtr_t, ytr).predict(Xte_t), results, cms)
    evaluate("Random Forest + TF-IDF", yte,
             RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1).fit(Xtr_t, ytr).predict(Xte_t),
             results, cms)

    # 4b. Word2Vec
    print("\n=== Word2Vec ===")
    w2v = Word2Vec([s.split() for s in Xtr], vector_size=args.w2v_dim,
                   window=5, min_count=2, workers=4, seed=SEED, epochs=20)
    Xtr_w = np.vstack([doc_vector(s, w2v, args.w2v_dim) for s in Xtr])
    Xte_w = np.vstack([doc_vector(s, w2v, args.w2v_dim) for s in Xte])
    evaluate("Logistic Regression + Word2Vec", yte,
             LogisticRegression(max_iter=1000, random_state=SEED).fit(Xtr_w, ytr).predict(Xte_w),
             results, cms)
    evaluate("SVM + Word2Vec", yte,
             SVC(kernel="rbf", C=1.0, random_state=SEED).fit(Xtr_w, ytr).predict(Xte_w),
             results, cms)
    evaluate("Naive Bayes + Word2Vec", yte,
             GaussianNB().fit(Xtr_w, ytr).predict(Xte_w), results, cms)
    evaluate("Random Forest + Word2Vec", yte,
             RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1).fit(Xtr_w, ytr).predict(Xte_w),
             results, cms)

    # 5. Simpan
    json.dump(results, open(os.path.join(args.outdir, "results_classic.json"), "w"), indent=2)
    json.dump(cms, open(os.path.join(args.outdir, "confusion_classic.json"), "w"), indent=2)
    print(f"\nHasil disimpan ke {args.outdir}/")


if __name__ == "__main__":
    main()
