# ============================================================================
#  CARA PAKAI:
#  - SEL PERTAMA  = blok di bawah penanda "### SEL 1 ###"
#  - SEL KEDUA    = blok di bawah penanda "### SEL 2 ###"
#  Buat 2 sel kode di Colab, copy-paste tiap blok ke selnya, jalankan berurutan.
#  Pastikan Runtime = GPU T4.
# ============================================================================


# ============================ ### SEL 1 ### =================================
# (instalasi + upload dataset — jalankan, lalu klik "Pilih File" unggah CSV)

!pip install -q transformers==4.44.2 datasets==2.21.0 scikit-learn pandas

from google.colab import files
uploaded = files.upload()


# ============================ ### SEL 2 ### =================================
# (seluruh pipeline: data -> tokenisasi -> latih IndoBERT -> hasil)
# Copy-paste SEMUA baris di bawah ini ke satu sel baru, lalu jalankan.

import pandas as pd, numpy as np, torch, random, re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# --- Deteksi nama file yang diunggah secara otomatis (tahan spasi/underscore) ---
fname = list(uploaded.keys())[0]
print("Memuat file:", fname)
df = pd.read_csv(fname, sep="\t")
df = df.drop_duplicates().dropna().reset_index(drop=True)

# Label asli: -1 negatif, 0 netral, 1 positif -> petakan ke 0,1,2
label_map = {-1: 0, 0: 1, 1: 2}
inv_map = {0: "Negatif", 1: "Netral", 2: "Positif"}
df["label"] = df["sentimen"].map(label_map)
print("Jumlah data:", len(df))
print(df["label"].value_counts())

# Pembersihan ringan (transformer tidak butuh stemming/stopword)
def light_clean(t):
    t = str(t)
    t = re.sub(r"http\S+|www\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
df["text"] = df["Tweet"].apply(light_clean)

# Split 70:30 stratified (sama seperti eksperimen ML klasik)
train_df, test_df = train_test_split(
    df[["text", "label"]], test_size=0.30, random_state=SEED, stratify=df["label"])
print(f"Train: {len(train_df)}  Test: {len(test_df)}")

# Tokenisasi
MODEL_NAME = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_ds = Dataset.from_pandas(train_df, preserve_index=False)
test_ds  = Dataset.from_pandas(test_df,  preserve_index=False)
def tok(b): return tokenizer(b["text"], truncation=True, padding="max_length", max_length=128)
train_ds = train_ds.map(tok, batched=True).rename_column("label", "labels")
test_ds  = test_ds.map(tok,  batched=True).rename_column("label", "labels")
cols = ["input_ids", "attention_mask", "labels"]
train_ds.set_format("torch", columns=cols)
test_ds.set_format("torch", columns=cols)

# Model + metrik
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f}

args = TrainingArguments(
    output_dir="./indobert_out", num_train_epochs=3,
    per_device_train_batch_size=16, per_device_eval_batch_size=32,
    learning_rate=2e-5, weight_decay=0.01,
    eval_strategy="epoch", save_strategy="no",
    logging_steps=50, seed=SEED, report_to="none")

trainer = Trainer(model=model, args=args,
                  train_dataset=train_ds, eval_dataset=test_ds,
                  compute_metrics=compute_metrics)

# Latih
trainer.train()

# Evaluasi akhir
metrics = trainer.evaluate()
acc  = metrics["eval_accuracy"]*100
prec = metrics["eval_precision"]*100
rec  = metrics["eval_recall"]*100
f1   = metrics["eval_f1"]*100

print("\n=========== HASIL INDOBERT (untuk Tabel 2 naskah) ===========")
print(f"Akurasi = {acc:.2f}")
print(f"Presisi = {prec:.2f}")
print(f"Recall  = {rec:.2f}")
print(f"F1      = {f1:.2f}")

# Confusion matrix
pred_out = trainer.predict(test_ds)
preds = np.argmax(pred_out.predictions, axis=1)
cm = confusion_matrix(pred_out.label_ids, preds, labels=[0,1,2])
print("\nConfusion matrix [baris=aktual, kolom=prediksi], urutan [Negatif, Netral, Positif]:")
print(cm)
print("\nKirim 4 angka di atas + matriks ini ke Claude untuk dimasukkan ke naskah.")
