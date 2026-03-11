# Sentiment Analysis - Ulasan Hotel
### Model: LSTM + Word2Vec | Metode: CRISP-DM

Proyek analisis sentimen ulasan hotel berbahasa Indonesia menggunakan Deep Learning (LSTM) dengan Word Embedding Word2Vec. Dataset diambil dari platform Traveloka melalui web scraping pada 12 hotel target.

---

## Hasil Akhir

| Metrik | Nilai |
|--------|-------|
| Val Accuracy | ~85% |
| Model | LSTM + Word2Vec |
| Total Data | 7.188 (balanced) |
| Kelas | Positif & Negatif |

---

## Struktur Proyek

```
├── Sentiment_Analysis.ipynb     # Notebook utama
├── dataset_preprocessed.csv     # Dataset setelah preprocessing
├── word2vec_hotel.model         # Model Word2Vec terlatih
├── lstm_sentiment_model.h5      # Model LSTM terlatih
├── Amalia Hotel Lampung.csv
├── Anugerah Express Hotel.csv
├── Anugrah Hotel Sukabumi.csv
├── Aryaduta Suite Semanggi.csv
├── Asoka Luxury Hotel Lampung.csv
├── Balcony Hotel Sukabumi.csv
├── Grand Anugerah Hotel.csv
├── Putri Duyung Ancol.csv
├── RedDoorz Syariah near Wisata Situ Gunung Sukabumi.csv
├── Sparks Odeon Sukabumi.csv
├── The Acacia Hotel Jakarta.csv
└── Grand Citihub Hotel @Kartini.csv
```

---

## Alur Pipeline

```
Raw CSV Files
     ↓
Load & Parse (fix format CSV non-standar)
     ↓
Labeling Sentimen (rating ≤ 6.9 = negatif, ≥ 7.0 = positif)
     ↓
Balancing Data (undersampling → 3.594 : 3.594)
     ↓
Preprocessing
  ├── Data Cleaning
  ├── Case Folding
  ├── Normalisasi Slang + Terjemahan Kata Inggris
  ├── Stopword Removal
  ├── Stemming (Sastrawi)
  └── Tokenisasi
     ↓
Word2Vec Embedding
     ↓
Split Data (80% train, 20% test)
     ↓
LSTM Model
     ↓
Evaluasi
```

---

## Dataset

- **Sumber**: Traveloka (web scraping)
- **Total raw**: 19.134 baris dari 12 hotel
- **Skala rating**: 1–10
- **Setelah labeling**: 15.540 positif, 3.594 negatif (imbalanced)
- **Setelah balancing**: 3.594 positif, 3.594 negatif = **7.188 baris**

### Distribusi Rating
- Rating ≤ 6.9 → **Negatif**
- Rating ≥ 7.0 → **Positif**
- Rating minimum di dataset: 4.5
- Rating terbanyak: 8.5 (4.861 ulasan)

---

## Preprocessing

| Tahap | Keterangan |
|-------|------------|
| Data Cleaning | Hapus emoji, simbol, URL, angka, karakter non-ASCII |
| Case Folding | Ubah ke huruf kecil |
| Normalisasi Slang | Kamus 30+ kata slang + 16 kata bahasa Inggris bermakna sentimen |
| Stopword Removal | NLTK bahasa Indonesia + kata umum bahasa Inggris (hotel, room, staff, service) |
| Pengecualian Stopword | Kata negasi tetap dipertahankan: tidak, kurang, bukan, belum, jangan, tanpa |
| Stemming | PySastrawi |
| Tokenisasi | Split per kata |

---

## Model LSTM

```
Embedding Layer   → vocab_size x 100 (bobot dari Word2Vec, frozen)
LSTM Layer        → 128 units
Dropout           → 0.5
Dense Layer       → 64 units, ReLU
Dropout           → 0.3
Output Layer      → 1 unit, Sigmoid (biner: 0=negatif, 1=positif)
```

### Hyperparameter

| Parameter | Nilai |
|-----------|-------|
| Optimizer | Adam (lr=0.0005) |
| Loss | Binary Crossentropy |
| Batch Size | 64 |
| Max Epochs | 20 |
| Early Stopping | patience=3, monitor=val_loss |
| MAX_WORDS | 10.000 |
| MAX_LEN | 100 |
| Embedding Dim | 100 |

---

## Word2Vec

| Parameter | Nilai |
|-----------|-------|
| vector_size | 100 |
| window | 5 |
| min_count | 2 |
| epochs | 10 |
| Vocab size | 3.363 kata |
| Coverage | 49.6% dari tokenizer vocab |

---

## Cara Menjalankan

### 1. Persiapan
```bash
pip install pandas numpy matplotlib seaborn gensim
pip install tensorflow nltk PySastrawi
```

### 2. Di Google Colab
- Upload semua file CSV ke Colab
- Pastikan Runtime menggunakan **GPU (T4)**
- Jalankan cell secara berurutan dari atas ke bawah

### 3. Jika Colab Restart
Jalankan cell **Load Backup** yang tersedia di awal notebook:
```python
df_balanced = pd.read_csv('dataset_preprocessed.csv')
df_balanced['tokens'] = df_balanced['review_clean'].apply(lambda x: str(x).split())
```

---

## Kendala yang Dihadapi

### 1. Format CSV Non-Standar
**Masalah**: Seluruh baris CSV terbungkus dalam satu tanda kutip besar, menyebabkan kolom `rating` dan `review` terbaca semua `NaN`.
```
# Format bermasalah di file:
"s***o,""10,0"",""Pelayanan cukup baik, akses parkir..."""
```
**Solusi**: Parsing manual per baris — kupas tanda kutip terluar, unescape `""` menjadi `"`, lalu parse ulang dengan `csv.reader`.

---

### 2. Rating Menggunakan Koma sebagai Desimal
**Masalah**: Rating ditulis `"10,0"` (koma) bukan `"10.0"` (titik), menyebabkan pandas salah membaca kolom.

**Solusi**:
```python
df['rating'] = df['rating'].str.replace(',', '.').astype(float)
```

---

### 3. Encoding Error
**Masalah**: Beberapa file CSV tidak menggunakan encoding UTF-8, menyebabkan `UnicodeDecodeError`.

**Solusi**:
```python
open(file, 'r', encoding='utf-8', errors='ignore')
```

---

### 4. Data Sangat Imbalanced
**Masalah**: Dataset sangat condong ke positif — 81.2% positif vs 18.8% negatif (rasio 1:4), karena mayoritas tamu hotel memberikan rating tinggi.

**Solusi**: Undersampling data positif agar seimbang 50:50.
```python
df_pos = df_labeled[df_labeled['sentiment']=='positive'].sample(n=jumlah_negatif, random_state=42)
```

---

### 5. Kata Negasi Terhapus oleh Stopword Removal
**Masalah**: Kata seperti `tidak`, `kurang`, `bukan` ikut terhapus padahal bermakna sentimen penting. Contoh: "kurang luas" menjadi "luas" yang berarti sebaliknya.

**Solusi**: Buat pengecualian stopwords untuk kata negasi:
```python
KATA_PENTING = {'tidak', 'kurang', 'bukan', 'belum', 'jangan', 'tanpa'}
STOPWORDS_ID = STOPWORDS_ID - KATA_PENTING
```

---

### 6. Model Tidak Belajar (Accuracy Stuck 50%)
**Masalah**: Accuracy stuck di 50% karena Embedding layer tidak menggunakan bobot Word2Vec — model belajar dari bobot random.

**Solusi**: Masukkan embedding matrix dari Word2Vec ke layer Embedding:
```python
Embedding(
    input_dim = vocab_size,
    output_dim = embedding_dim,
    weights   = [embedding_matrix],  # bobot Word2Vec
    trainable = False                # freeze embedding
)
```

---

### 7. Training Tidak Stabil
**Masalah**: Akurasi naik turun drastis antar epoch karena learning rate default Adam (0.001) terlalu besar.

**Solusi**: Turunkan learning rate:
```python
Adam(learning_rate=0.0005)
```

---

### 8. Kata Bahasa Inggris dalam Review
**Masalah**: 31% review mengandung kata bahasa Inggris (good, bad, noisy, dll) yang tidak dikenali Sastrawi.

**Solusi**: Tambahkan terjemahan ke kamus normalisasi slang:
```python
'good': 'bagus', 'bad': 'buruk', 'noisy': 'berisik', ...
```
Dan tambahkan kata umum tidak bermakna sentimen ke stopwords:
```python
STOPWORDS_TAMBAHAN = {'hotel', 'room', 'staff', 'service'}
```

---

## Library yang Digunakan

| Library | Kegunaan |
|---------|----------|
| pandas | Manipulasi data |
| numpy | Komputasi numerik |
| matplotlib, seaborn | Visualisasi |
| gensim | Word2Vec |
| tensorflow/keras | LSTM |
| nltk | Stopword bahasa Indonesia |
| PySastrawi | Stemming bahasa Indonesia |
| scikit-learn | Split data, evaluasi |
| tqdm | Progress bar |

---

## Referensi Metode
- CRISP-DM (Cross-Industry Standard Process for Data Mining)
- Word2Vec: Mikolov et al. (2013)
- LSTM: Hochreiter & Schmidhuber (1997)
- PySastrawi: Algoritma stemming bahasa Indonesia
