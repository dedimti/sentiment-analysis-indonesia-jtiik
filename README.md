# Analisis Sentimen Bahasa Indonesia: Perbandingan Model Klasik dan IndoBERT

Repositori ini berisi kode, hasil, dan figur untuk penelitian perbandingan
analisis sentimen berbahasa Indonesia pada media sosial Twitter. Penelitian
membandingkan **empat algoritma klasik** (Logistic Regression, SVM, Naive Bayes,
Random Forest) dengan **dua representasi fitur** (TF-IDF dan Word2Vec) — total
delapan kombinasi — serta model berbasis transformer **IndoBERT** sebagai
pembanding.

Naskah disusun mengikuti gaya **JTIIK** (Jurnal Teknologi Informasi dan Ilmu
Komputer, Universitas Brawijaya).

## Dataset

Penelitian menggunakan **Indonesian Sentiment Twitter Dataset** (Ferdiana dkk.,
2019): 10.806 tweet berbahasa Indonesia berlabel tiga kelas (positif, negatif,
netral).

> **Dataset tidak disertakan** dalam repositori ini sesuai ketentuan
> penggunaannya. Unduh dari sumber asli dan sitir publikasinya:
>
> Ferdiana, R., Jatmiko, F., Purwanti, D.D., Ayu, A.S.T., & Dicka, W.F. (2019).
> *Dataset Indonesia untuk Analisis Sentimen.* JNTETI, 8(4), 334–339.
> https://doi.org/10.22146/jnteti.v8i4.533

Letakkan file `Indonesian_Sentiment_Twitter_Dataset_Labeled.csv` (tab-separated,
kolom `sentimen` dan `Tweet`) di folder `data/`.

## Struktur Repositori

```
.
├── src/
│   ├── run_classic.py      # Pipeline 8 model klasik (preprocessing → fitur → model → evaluasi)
│   └── make_figures.py     # Membuat figur dari hasil JSON
├── notebooks/
│   └── indobert_colab.py   # Fine-tuning IndoBERT (jalankan di Google Colab dengan GPU)
├── results/
│   ├── results_classic.json    # Metrik 8 model klasik
│   ├── confusion_classic.json  # Confusion matrix model klasik
│   └── results_indobert.json   # Metrik & confusion matrix IndoBERT
├── figures/
│   ├── alur_metode.png
│   ├── perbandingan.png
│   ├── confusion_lr_tfidf.png
│   └── confusion_indobert.png
├── requirements.txt
├── LICENSE
└── README.md
```

## Cara Menjalankan

### 1. Model klasik

```bash
pip install -r requirements.txt
python src/run_classic.py --data data/Indonesian_Sentiment_Twitter_Dataset_Labeled.csv --outdir results
```

> Catatan: tahap stemming Sastrawi pada 10 ribu tweet memakan waktu beberapa
> menit (stemming di-cache per kata unik untuk efisiensi).

### 2. IndoBERT (Google Colab, GPU)

Buka `notebooks/indobert_colab.py` di Google Colab, set **Runtime → GPU (T4)**,
lalu jalankan. Skrip otomatis mendeteksi nama file dataset yang diunggah.

### 3. Membuat figur

```bash
python src/make_figures.py --results results/results_classic.json \
    --confusion results/confusion_classic.json --outdir figures
```

## Ringkasan Hasil

Evaluasi pada 3.075 data uji (split 70:30 terstratifikasi, seed 42):

| Model | Fitur | Akurasi | Presisi | Recall | F1 |
|---|---|---|---|---|---|
| **IndoBERT** | Fine-tuned | **69,40** | 68,13 | 67,66 | **67,88** |
| Logistic Regression | TF-IDF | 60,16 | 58,60 | 52,73 | 53,78 |
| SVM | TF-IDF | 59,41 | 56,81 | 52,66 | 53,53 |
| Naive Bayes | TF-IDF | 58,18 | 61,25 | 47,08 | 46,91 |
| Random Forest | TF-IDF | 57,66 | 55,53 | 49,71 | 50,40 |
| Logistic Regression | Word2Vec | 53,95 | 51,36 | 43,27 | 42,41 |
| SVM | Word2Vec | 53,43 | 53,52 | 40,17 | 36,53 |
| Random Forest | Word2Vec | 51,45 | 46,51 | 42,23 | 41,77 |
| Naive Bayes | Word2Vec | 44,07 | 42,70 | 42,73 | 42,12 |

Temuan utama: IndoBERT unggul secara keseluruhan; di antara model klasik,
TF-IDF konsisten mengungguli Word2Vec, dengan Logistic Regression + TF-IDF
sebagai kombinasi klasik terbaik.

## Konfigurasi Reproducibility

- Split: 70:30, terstratifikasi, `random_state=42`
- TF-IDF: `max_features=5000`, `ngram_range=(1,2)`, `min_df=2`
- Word2Vec: Skip-gram, `vector_size=100`, `window=5`, `min_count=2`, `epochs=20`
- Random Forest: `n_estimators=200`; Logistic Regression: `max_iter=1000`
- IndoBERT: `indobenchmark/indobert-base-p1`, max_length=128, batch=16, lr=2e-5, 3 epoch

## Model & Lisensi

IndoBERT (`indobenchmark/indobert-base-p1`) berlisensi MIT. Bila menggunakannya,
sitir:

> Wilie, B., Vincentio, K., Winata, G.I., Cahyawijaya, S., dkk. (2020).
> *IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language
> Understanding.* Proceedings of AACL-IJCNLP 2020.

Kode dalam repositori ini dirilis di bawah lisensi MIT (lihat `LICENSE`).
