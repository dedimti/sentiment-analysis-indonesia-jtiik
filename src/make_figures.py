"""
Membuat figur untuk naskah dari file hasil JSON.
Cara pakai:
    python src/make_figures.py --results results/results_classic.json \
        --confusion results/confusion_classic.json --outdir figures
Catatan: hasil IndoBERT (akurasi 69,40; F1 67,88) dan confusion matrix-nya
diperoleh dari notebook notebooks/indobert_colab.py dan dimasukkan manual di sini.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Hasil IndoBERT dari fine-tuning (lihat notebooks/indobert_colab.py)
INDOBERT = {"accuracy": 69.40, "precision": 68.13, "recall": 67.66, "f1": 67.88}
INDOBERT_CM = [[498, 227, 82], [211, 1144, 171], [53, 198, 494]]


def fig_comparison(res, outdir):
    res = dict(res)
    res["IndoBERT"] = INDOBERT
    order = ["IndoBERT",
             "Logistic Regression + TF-IDF", "SVM + TF-IDF",
             "Naive Bayes + TF-IDF", "Random Forest + TF-IDF",
             "Logistic Regression + Word2Vec", "SVM + Word2Vec",
             "Naive Bayes + Word2Vec", "Random Forest + Word2Vec"]
    short = ["IndoBERT", "LR\nTF-IDF", "SVM\nTF-IDF", "NB\nTF-IDF", "RF\nTF-IDF",
             "LR\nW2V", "SVM\nW2V", "NB\nW2V", "RF\nW2V"]
    acc = [res[m]["accuracy"] for m in order]
    f1 = [res[m]["f1"] for m in order]
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = ["#2C5F2D"] + ["#34699A"] * 8
    b1 = ax.bar(x - w/2, acc, w, label="Akurasi", color=colors)
    b2 = ax.bar(x + w/2, f1, w, label="F1-score (makro)", color="#C7522A")
    ax.set_ylabel("Persentase (%)"); ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=8); ax.set_ylim(0, 80)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for b in list(b1) + list(b2):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                f"{b.get_height():.1f}", ha="center", fontsize=6.5)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "perbandingan.png"), dpi=200, bbox_inches="tight")
    plt.close()


def fig_confusion(cm, title, fname, outdir, cmap="Blues"):
    cm = np.array(cm); labels = ["Negatif", "Netral", "Positif"]
    fig, ax = plt.subplots(figsize=(5, 4.3))
    im = ax.imshow(cm, cmap=cmap)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
    th = cm.max() / 2
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > th else "black", fontsize=12)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname), dpi=200, bbox_inches="tight")
    plt.close()


def fig_flow(outdir):
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(4.2, 8.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 20); ax.axis("off")
    steps = ["Pengumpulan Data\n(10.806 tweet berlabel)",
             "Preprocessing\n(case folding, cleaning,\ntokenisasi, stopword, stemming)",
             "Pembagian Data\n(latih 70% : uji 30%)",
             "Ekstraksi Fitur\nTF-IDF  |  Word2Vec",
             "Pelatihan Model\nLR, SVM, NB, RF\n(+ IndoBERT)",
             "Evaluasi\n(akurasi, presisi,\nrecall, F1-score)",
             "Analisis &\nPerbandingan Hasil"]
    colors = ["#E8EEF4", "#D6E2EE", "#C4D6E8", "#B2CAE2", "#A0BEDC", "#8EB2D6", "#7CA6D0"]
    y, h, gap = 18.5, 2.0, 0.55
    for i, (s, c) in enumerate(zip(steps, colors)):
        ax.add_patch(FancyBboxPatch((2, y-h), 6, h,
                     boxstyle="round,pad=0.1,rounding_size=0.25",
                     linewidth=1.2, edgecolor="#34699A", facecolor=c))
        ax.text(5, y-h/2, s, ha="center", va="center", fontsize=8.5)
        if i < len(steps)-1:
            ax.add_patch(FancyArrowPatch((5, y-h), (5, y-h-gap),
                         arrowstyle="-|>", mutation_scale=14,
                         color="#34699A", linewidth=1.3))
        y -= (h + gap)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "alur_metode.png"), dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results_classic.json")
    ap.add_argument("--confusion", default="results/confusion_classic.json")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    res = json.load(open(args.results))
    cms = json.load(open(args.confusion))

    fig_flow(args.outdir)
    fig_comparison(res, args.outdir)
    fig_confusion(cms["Logistic Regression + TF-IDF"],
                  "LR + TF-IDF", "confusion_lr_tfidf.png", args.outdir, "Blues")
    fig_confusion(INDOBERT_CM, "IndoBERT", "confusion_indobert.png", args.outdir, "Greens")
    print(f"Gambar disimpan ke {args.outdir}/")


if __name__ == "__main__":
    main()
