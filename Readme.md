# 📄 Laporan Proyek: Sistem Rekomendasi Buku

## Judul & Identitas

**Judul:** Book Recommendation System
**Nama:** Deawi Guna Pratiwi
**Email:** [deawigunapratiwi@gmail.com](mailto:deawigunapratiwi@gmail.com)
**Sumber Dataset:** [Book-Crossing: User review ratings - Kaggle](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset)
**Jumlah data:** 1.031.175 baris data

---

## 1. 🏢 Project Overview

Di era digital, informasi buku yang melimpah justru menjadi tantangan bagi pembaca dalam menemukan bacaan yang sesuai dengan preferensi mereka. Oleh karena itu, diperlukan sebuah sistem rekomendasi buku yang mampu menyarankan judul-judul yang relevan secara personal.

Dataset yang digunakan adalah **Book-Crossing: User review ratings** dari Kaggle. Dataset ini memuat lebih dari 1 juta data interaksi pengguna (user-book-rating), metadata buku, dan informasi dasar pengguna.

Tujuan proyek ini:

* Membangun sistem rekomendasi berbasis konten (Content-Based Filtering).
* Menganalisis data pengguna dan buku secara eksploratif.
* Menyajikan rekomendasi buku yang personal dan relevan berdasarkan metadata dan sinopsis.

---

## 2. 🔍 Business Understanding

### 🔎 Problem Statements

1. Pengguna kesulitan menemukan buku sesuai preferensi karena tidak ada sistem rekomendasi.
2. Data pengguna dan metadata buku belum dimanfaatkan optimal.
3. Belum ada analisis mendalam terhadap interaksi pengguna dan karakteristik buku.

### 🎯 Goals

1. Membangun sistem rekomendasi berbasis konten.
2. Melakukan eksplorasi data untuk memahami struktur dan hubungan antar variabel.
3. Menghasilkan rekomendasi berdasarkan kesamaan konten buku.

### 🛠️ Solution Approach

* Melakukan EDA untuk memahami pola distribusi data.
* Melakukan pembersihan data (handling missing, outlier, duplikasi).
* Membangun sistem rekomendasi menggunakan TF-IDF + cosine similarity.

---

## 3. 📊 Data Understanding

Dataset yang digunakan adalah hasil pembersihan awal dengan nama **Preprocessed\_data.csv**. Dataset ini merupakan hasil dari penggabungan dan pembersihan beberapa file mentah dari Kaggle: [Book-Crossing Dataset](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset).

### Struktur Dataset Preprocessed:

Dataset `Preprocessed_data.csv` merupakan gabungan awal dari tiga file mentah berikut:

* Metadata buku (`BX-Books.csv`)
* Data pengguna (`BX-Users.csv`)
* Data interaksi (`BX-Book-Ratings.csv`)

**Ukuran Dataset Awal:**

* Total baris: 1.031.175
* Total kolom: 19

**Kondisi Awal Dataset:**

* Dataset masih mengandung nilai aneh dan format yang belum terstandardisasi.
* Kolom `year_of_publication` masih bertipe string dan memuat nilai tidak logis (seperti 0 atau >2025).
* Kolom `Category` masih berupa string literal dari list, misalnya `'[Fiction]'`.
* Beberapa ringkasan (`Summary`) memiliki nilai tidak valid (seperti angka '9') atau terlalu pendek.
* Terdapat entri duplikat yang belum dibersihkan berdasarkan kombinasi judul dan penerbit.

**Jumlah Missing Values:**

```
user_id                     0
location                   0
age                        0
isbn                       0
rating                     0
book_title                 0
book_author                1
year_of_publication        0
publisher                  0
img_s                      0
img_m                      0
img_l                      0
Summary                    0
Language                   0
Category                   0
city                   14095
state                  22767
country                35365
```

**Kolom yang digunakan:**

* `user_id`, `book_title`, `Summary`, `Category`, `publisher`, `year_of_publication`, `Language`, `rating`, `age`

* `user_id`, `book_title`, `Summary`, `Category`, `publisher`, `year_of_publication`, `Language`, `rating`, `age`

* **Total baris mentah:** 1.031.175 interaksi rating.

* **Missing values:** ditemukan pada kolom `age`, `publisher`, dan `Summary`.

* **Outliers:** nilai umur <10 dan >90 dianggap outlier dan dibersihkan.

* **Duplikasi:** data duplikat dihapus berdasarkan ISBN dan user\_id.

### Penjelasan Fitur:

| Nama Fitur            | Deskripsi             | Status                                       |
| --------------------- | --------------------- | -------------------------------------------- |
| `user_id`             | ID pengguna unik      | Digunakan                                    |
| `location`            | Lokasi pengguna       | Tidak digunakan (informasi terlalu granular) |
| `age`                 | Umur pengguna         | Digunakan (outlier dibersihkan)              |
| `isbn`                | ISBN buku             | Digunakan sebagai identitas unik             |
| `book_title`          | Judul buku            | Digunakan                                    |
| `book_author`         | Penulis               | Tidak digunakan                              |
| `year_of_publication` | Tahun terbit          | Digunakan (beberapa nilai dibersihkan)       |
| `publisher`           | Penerbit              | Digunakan                                    |
| `Summary`             | Ringkasan konten buku | Digunakan                                    |
| `Category`            | Kategori buku         | Digunakan                                    |
| `Language`            | Bahasa buku           | Digunakan (nilai anomali dihapus)            |
| `rating`              | Rating user           | Digunakan untuk validasi relevansi           |

---

## 4. 🧹 Data Preparation

### Langkah-Langkah:

1. **Menghapus kolom tidak relevan:** seperti `Unnamed: 0` dan URL gambar.
2. **Handling missing values:**

   * Kolom `Summary` dan `Category` yang kosong dihapus.
   * Tahun publikasi dengan nilai aneh ("0", "nan", dst) disesuaikan jika memungkinkan.
3. **Handling outlier:**

   * Umur <10 atau >90 dihapus.
4. **Encoding fitur teks:**

   * Menggabungkan kolom `book_title`, `Summary`, `Category`, dan `publisher` ke kolom baru `combined`.
   * Melakukan **feature extraction** menggunakan **TF-IDF Vectorizer** terhadap kolom `combined`.

Dengan langkah ini, seluruh fitur yang digunakan dalam pemodelan sudah dalam format numerik yang bisa digunakan oleh algoritma machine learning.

---

## 💡 5. Modelling: Content-Based Filtering

### 1. Definisi & Cara Kerja Content-Based Filtering

Content-based filtering bekerja dengan mencari kemiripan antar item berdasarkan fitur-fitur deskriptif (fitur konten) dari item itu sendiri. Dalam konteks ini, fitur-fitur konten adalah `Summary`, `Category`, `book_title`, dan `publisher`.

### 2. Algoritma yang Digunakan

* **TF-IDF (Term Frequency-Inverse Document Frequency)**: digunakan untuk mengubah teks mentah menjadi representasi vektor numerik yang menonjolkan kata-kata penting.
* **Cosine Similarity**: digunakan untuk mengukur seberapa mirip dua vektor teks.
* **Nearest Neighbors**: digunakan untuk menemukan buku dengan kemiripan tertinggi terhadap buku referensi berdasarkan nilai cosine similarity.

### 3. Fungsi Rekomendasi:

Fungsi `recommend_books_nn(title, top_n)` menerima judul buku sebagai input dan mengembalikan Top-N buku paling mirip berdasarkan cosine similarity.

### 4. Contoh Output:

**Input**: *The Secret Life of Bees*
**Rekomendasi**: *Black Boy*, *Midnight Heat*, *Clover*, dll.

**Input**: *A Painted House*
**Rekomendasi**: *Boy of the Painted Cave*, *River of Earth*, *The Wedding Dress*, dll.

---

## 📊 6. Evaluasi Sistem Rekomendasi

### Evaluasi Kuantitatif: Precision\@5

Evaluasi dilakukan menggunakan metrik **Precision\@5**, yang mengukur proporsi item relevan di antara 5 item teratas yang direkomendasikan. Evaluasi dilakukan langsung di notebook dengan pendekatan sebagai berikut:

```python
recommended_df = recommend_books_nn("A Painted House", top_n=5)
recommended_books = recommended_df['book_title'].tolist()
ground_truth_books = [
    "A Painted House",
    "A Painted House (Limited Edition)",
    "Boy of the Painted Cave",
    "River of Earth",
    "The Wedding Dress"
]
def precision_at_k(recommendations, ground_truth, k):
    if not ground_truth:
        return 0.0
    recommended_top_k = recommendations[:k]
    relevant_and_recommended = set(recommended_top_k) & set(ground_truth)
    return len(relevant_and_recommended) / k
precision = precision_at_k(recommended_books, ground_truth_books, k=5)
print(f"Precision@5: {precision:.2f}")
```

**Hasil:** Precision\@5 = **1.00**, yang berarti seluruh hasil rekomendasi dianggap relevan menurut ground truth. Ini menunjukkan bahwa sistem berhasil merekomendasikan buku yang sesuai secara konten dan judul.

---
## 7. Struktur Laporan
* **1. Project Overview.**
* **2. Business Understanding**
* **3. Data Understanding**
* **4. Data Preparation**
* **5. Modelling and Results**
* **6. Evaluation**
---

## 📍 Kesimpulan

Sistem rekomendasi buku berbasis content-based filtering yang dibangun dalam proyek ini mampu:

* Memanfaatkan konten buku (summary, genre, judul, penerbit) untuk mengukur kemiripan antar buku.
* Memberikan saran buku yang relevan secara tema dan struktur cerita.
* Memberikan alternatif yang baik untuk situasi cold-start (buku baru).

Namun, untuk meningkatkan performa dan skalabilitas sistem, disarankan untuk mengintegrasikan pendekatan hybrid dan memasukkan umpan balik eksplisit pengguna.

---

## 🗓️ Referensi

1. [Book-Crossing Dataset - Kaggle](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset)
2. Ricci, F., Rokach, L., & Shapira, B. (2011). *Introduction to Recommender Systems Handbook*.
3. Scikit-learn documentation: [https://scikit-learn.org](https://scikit-learn.org)
4. TF-IDF: Term Frequency-Inverse Document Frequency method
5. Cosine Similarity and Nearest Neighbors algorithm (sklearn)
