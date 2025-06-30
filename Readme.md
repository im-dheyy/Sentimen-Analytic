# 📄 Laporan Proyek: Sistem Rekomendasi Buku

## 1. Judul & Identitas

**Judul:** Book Recommendation System
**Nama:** Deawi Guna Pratiwi
**Email:** [deawigunapratiwi@gmail.com](mailto:deawigunapratiwi@gmail.com)
**Sumber Dataset:** [Book-Crossing: User review ratings - Kaggle](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset)
**Jumlah data:** 1.031.175 baris data

---

## 2. 🏢 Proyek Overview

Di era digital, informasi buku yang melimpah justru menjadi tantangan bagi pembaca dalam menemukan bacaan yang sesuai dengan preferensi mereka. Oleh karena itu, diperlukan sebuah sistem rekomendasi buku yang mampu menyarankan judul-judul yang relevan secara personal.

Dataset yang digunakan adalah **Book-Crossing: User review ratings** dari Kaggle. Dataset ini memuat lebih dari 1 juta data interaksi pengguna (user-book-rating), metadata buku, dan informasi dasar pengguna.

Tujuan proyek ini:

* Membangun sistem rekomendasi berbasis konten (Content-Based Filtering).
* Menganalisis data pengguna dan buku secara eksploratif.
* Menyajikan rekomendasi buku yang personal dan relevan berdasarkan metadata dan sinopsis.

---

## 3. 🔍 Business Understanding

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

## 4. 📊 Data Understanding

Dataset terdiri dari file "Preprocessed\_data.csv" yang telah dibersihkan. Beberapa fitur penting:

* `book_title`, `Summary`, `Category`, `publisher` (fitur konten buku).
* `user_id`, `age`, `location` (informasi pengguna).
* `rating` (interaksi pengguna).

Total data awal: **1.031.175 baris**, 19 kolom.

Langkah awal:

* Menghapus kolom tidak relevan (`Unnamed: 0`).
* Membersihkan nilai outlier pada usia (<10 atau >90).
* Mengimputasi tahun publikasi yang tidak valid.
* Parsing kategori dari string menjadi list.
* Menghapus ringkasan yang tidak valid dan entri duplikat.

---

## 📊 EDA (Exploratory Data Analysis)

### 1. Distribusi Rating

* Mayoritas rating adalah **0** (rating kosong/implisit).
* Rating eksplisit cenderung bernilai tinggi (7–10).
![Distribusi Nilai Rating](https://github.com/im-dheyy/Sentimen-Analytic/raw/main/Gambar/distribusinilairating.png)

### 2. Buku dengan Rating Terbanyak

* Buku seperti *Wild Animus*, *The Lovely Bones*, dan *The Da Vinci Code* paling banyak dirating.
![Top 10 Buku dengan Rating Terbanyak](https://github.com/im-dheyy/Sentimen-Analytic/raw/main/Gambar/top10bukudenganratingterbanyak.png)


### 3. Distribusi Usia Pengguna

* Dominasi pengguna pada rentang usia 30–40 tahun.
![Distribusi Usia Pengguna](https://github.com/im-dheyy/Sentimen-Analytic/raw/main/Gambar/distribusiusiapengguna.png)

### 4. Bahasa Buku Terpopuler

* Mayoritas buku berbahasa Inggris (`en`).
* Terdapat nilai anomali seperti "9" pada kolom bahasa.
![Top 10 Bahasa Buku Terpopuler](https://github.com/im-dheyy/Sentimen-Analytic/raw/main/Gambar/top10bahasabukuterpopuler.png)

### 5. Kategori Buku Terpopuler

* Kategori "Fiction" mendominasi.
* Diikuti oleh "Juvenile Fiction", "Biography", dll
![Top 10 Kategori Buku](https://github.com/im-dheyy/Sentimen-Analytic/blob/raw/main/Gambar/top10kategoribuku.png)

---

## 💡 Modelling: Content-Based Filtering

### 1. Tahapan Persiapan:

* Menggabungkan fitur teks: `Summary`, `Category`, `book_title`, `publisher` ke dalam 1 kolom `combined`.
* Menggunakan **TF-IDF Vectorizer** untuk mengubah teks menjadi vektor numerik.
* Membangun model **NearestNeighbors** dengan metrik cosine similarity.

### 2. Fungsi Rekomendasi:

Fungsi `recommend_books_nn(title, top_n)` menerima judul buku sebagai input dan mengembalikan Top-N buku paling mirip.

### 3. Contoh Output:

**Input**: *The Secret Life of Bees*
**Rekomendasi**: *Black Boy*, *Midnight Heat*, *Clover*, dll.

**Input**: *A Painted House*
**Rekomendasi**: *Boy of the Painted Cave*, *River of Earth*, *The Wedding Dress*, dll.

---

## 📊 Evaluasi Sistem Rekomendasi

### ✅ 1. Evaluasi Kualitatif

* Menggunakan review manual untuk mengevaluasi rekomendasi.
* Rekomendasi yang dihasilkan **relevan secara tematik dan emosional**.

### ✅ 2. Evaluasi Konsistensi (Top-N Overlap)

* Buku dengan judul mirip (*A Painted House* dan *A Painted House (Limited Edition)*) menghasilkan rekomendasi yang konsisten.

### ✅ 3. Evaluasi Coverage

* Dari lebih dari 1 juta data, tersisa \~96.000 data unik yang siap digunakan untuk sistem rekomendasi.

### ❌ Keterbatasan:

| Aspek                  | Keterangan                                    |
| ---------------------- | --------------------------------------------- |
| Tanpa rating eksplisit | Tidak bisa hitung Precision\@K atau Recall\@K |
| Tidak ada ground truth | Tidak ada label relevansi manual              |
| Cold-start user        | Tidak mempertimbangkan histori pengguna       |

### 💡 Rekomendasi Lanjutan:

* Bangun model **hybrid** (content + collaborative).
* Tambahkan **user feedback loop**.
* Buat dataset validasi untuk uji kuantitatif.
* Visualisasi sistem ke dalam dashboard interaktif (mis. Streamlit).

---

## 🔹 Kesimpulan

Sistem rekomendasi buku berbasis content-based filtering yang dibangun dalam proyek ini mampu:

* Memanfaatkan konten buku (summary, genre, judul, penerbit) untuk mengukur kemiripan antar buku.
* Memberikan saran buku yang relevan secara tema dan struktur cerita.
* Memberikan alternatif yang baik untuk situasi cold-start (buku baru).

Namun, untuk meningkatkan performa dan skalabilitas sistem, disarankan untuk mengintegrasikan pendekatan hybrid dan memasukkan umpan balik eksplisit pengguna.

---

## 📅 Referensi

1. [Book-Crossing Dataset - Kaggle](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset)
2. Ricci, F., Rokach, L., & Shapira, B. (2011). *Introduction to Recommender Systems Handbook*.
3. Scikit-learn documentation: [https://scikit-learn.org](https://scikit-learn.org)
4. TF-IDF: Term Frequency-Inverse Document Frequency method
5. Cosine Similarity and Nearest Neighbors algorithm (sklearn)
