"""
Eksperimen SMOTE: kinerja model klasik (TF-IDF) sebelum vs sesudah penyeimbangan kelas.
Cara pakai:
    python src/run_smote.py --data data/Indonesian_Sentiment_Twitter_Dataset_Labeled.csv
"""
import argparse, json, os, re
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

SEED=42; np.random.seed(SEED)
def clean(t):
    t=str(t).lower(); t=re.sub(r"http\S+|www\S+"," ",t); t=re.sub(r"@\w+|#\w+"," ",t)
    t=re.sub(r"[^a-z\s]"," ",t); t=re.sub(r"(.)\1{2,}",r"\1",t); return re.sub(r"\s+"," ",t).strip()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",required=True); ap.add_argument("--outdir",default="results")
    a=ap.parse_args(); os.makedirs(a.outdir,exist_ok=True)
    df=pd.read_csv(a.data,sep="\t").drop_duplicates().dropna().reset_index(drop=True)
    st=StemmerFactory().create_stemmer(); sw=set(StopWordRemoverFactory().get_stop_words()); cache={}
    def pre(x):
        out=[]
        for w in clean(x).split():
            if w in sw or len(w)<=1: continue
            if w not in cache: cache[w]=st.stem(w)
            out.append(cache[w])
        return " ".join(out)
    df["proc"]=df["Tweet"].apply(pre); df=df[df["proc"].str.strip()!=""].reset_index(drop=True)
    X,y=df["proc"].values,df["sentimen"].values
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=SEED,stratify=y)
    tf=TfidfVectorizer(max_features=5000,ngram_range=(1,2),min_df=2)
    Xtr_t=tf.fit_transform(Xtr); Xte_t=tf.transform(Xte)
    Xtr_sm,ytr_sm=SMOTE(random_state=SEED).fit_resample(Xtr_t,ytr)
    models={"Logistic Regression":LogisticRegression(max_iter=1000,random_state=SEED),
            "SVM":SVC(kernel="linear",random_state=SEED),
            "Naive Bayes":MultinomialNB(),
            "Random Forest":RandomForestClassifier(n_estimators=200,random_state=SEED,n_jobs=-1)}
    res={}
    for n,c in models.items():
        c.fit(Xtr_sm,ytr_sm); yp=c.predict(Xte_t)
        acc=accuracy_score(yte,yp); p,r,f,_=precision_recall_fscore_support(yte,yp,average="macro",zero_division=0)
        res[n]={"accuracy":round(acc*100,2),"precision":round(p*100,2),"recall":round(r*100,2),"f1":round(f*100,2)}
        print(f"{n:22s} acc={acc*100:.2f} f1={f*100:.2f}")
    json.dump(res,open(os.path.join(a.outdir,"smote_results.json"),"w"),indent=2)
    print("Disimpan ke",a.outdir)

if __name__=="__main__": main()
