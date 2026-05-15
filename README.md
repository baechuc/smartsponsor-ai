# 🎯 SponsorSmart AI

Sistem Pendukung Keputusan Penilaian Kelayakan Proposal Sponsorship menggunakan IndoBERT + SVM.

## 📁 Struktur Repo

```
├── capstone_kaggle.ipynb     # Notebook utama (training di Kaggle)
├── sponsorsmart_app.py       # Streamlit web app
├── requirements.txt          # Dependensi Python
└── README.md
```

## 🚀 Cara Menjalankan

### Training Model (Kaggle)
1. Upload dataset PDF ke **Kaggle Datasets** dengan nama `sponsorsmart-dataset`
2. Import `capstone_kaggle.ipynb` ke Kaggle Notebook
3. Aktifkan **GPU T4** di Settings → Accelerator
4. Run All → download `best_indobert_model/` dan `svm_model.pkl` dari tab Output

### Streamlit App (Lokal)
```bash
pip install -r requirements.txt
streamlit run sponsorsmart_app.py
```

Pastikan folder `best_indobert_model/` dan `svm_model.pkl` ada di direktori yang sama.

## 🔬 Variabel Rubric

| Variabel | Deskripsi |
|---|---|
| 📡 Exposure | Jangkauan audiens & media |
| 🎯 Relevansi | Kesesuaian dengan brand sponsor |
| 🎁 Benefit | Keuntungan nyata untuk sponsor |
| 💰 Anggaran | Kejelasan biaya & paket |
| 🏛️ Kredibilitas | Profesionalitas penyelenggara |

**Keputusan:** Total Skor ≥ 3 → **Layak**

## 🧠 Model

- **IndoBERT** (`indobenchmark/indobert-base-p1`) — fine-tuned untuk klasifikasi proposal
- **SVM + TF-IDF** — baseline model

> ⚠️ File model tidak disertakan di repo karena ukurannya besar. Jalankan notebook untuk melatih model.
