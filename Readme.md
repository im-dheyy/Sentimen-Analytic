# 📄 Laporan Proyek: Sistem Rekomendasi Buku

## 1. Judul & Identitas

**Judul:** Book Recommendation System
**Nama:** Deawi Guna Pratiwi
**Email:** [deawigunapratiwi@gmail.com](mailto:deawigunapratiwi@gmail.com)
**Sumber Dataset:** [Book-Crossing: User review ratings - Kaggle](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset)
**Jumlah data:** 1.031.175 baris data

---

## 2. Proyek Overview

Di era digital, informasi mengenai buku sangat melimpah. Meskipun hal ini memberikan banyak pilihan bagi pembaca, pada saat yang sama muncul tantangan besar dalam menemukan buku yang benar-benar relevan dengan minat dan preferensi masing-masing individu.

Untuk mengatasi permasalahan ini, proyek ini membangun sebuah **sistem rekomendasi buku** berbasis data yang mampu menyarankan buku secara personal berdasarkan data perilaku dan preferensi pengguna sebelumnya.

---

## 3. Business Understanding

Tujuan bisnis dari proyek ini adalah:

* Membantu pengguna menemukan buku yang relevan dan sesuai minat.
* Meningkatkan pengalaman pengguna dalam menjelajahi koleksi buku.
* Meningkatkan interaksi dan engagement pengguna terhadap platform penyedia buku.

### Problem Statements:

* Bagaimana menyusun sistem rekomendasi berdasarkan minat pengguna?
* Apa pendekatan terbaik antara content-based dan collaborative filtering?

---

## 4. Data Understanding

Dataset yang digunakan memiliki 3 bagian utama:

* **Books.csv**: informasi tentang buku (ISBN, judul, penulis, publisher, tahun terbit, dll).
* **Users.csv**: data tentang pengguna (user id, lokasi, usia).
* **Ratings.csv**: berisi rating pengguna terhadap buku (user id, ISBN, rating).

### Ringkasan:

* Jumlah pengguna: 278.858
* Jumlah buku: 271.379
* Jumlah rating: 1.031.175

Data perlu dibersihkan dari nilai yang hilang, tidak relevan, dan hanya mempertimbangkan rating eksplisit (> 0).

---

## 5. Exploratory Data Analysis (EDA)

EDA dilakukan untuk memahami karakteristik data, mendeteksi pola, dan anomali. Visualisasi dibuat untuk memberikan insight yang lebih mendalam terhadap distribusi data.

### Distribusi Rating Buku

* Mayoritas rating buku adalah 0 (implicit rating), sehingga difokuskan pada rating eksplisit.

![Distribusi Nilai Rating](https://github.com/im-dheyy/Sentimen-Analytic/raw/main/Gambar/distribusinilairating.png)

### Distribusi Usia Pengguna

* Mayoritas pengguna berada dalam rentang usia 20-40 tahun.
* Visualisasi: [distribusiusiapengguna.png](https://github.com/im-dheyy/Sentimen-Analytic/blob/main/Gambar/distribusiusiapengguna.png)

### Top 10 Buku Berdasarkan Popularitas Judul

* Buku-buku populer terdeteksi melalui jumlah rating terbanyak.
* Visualisasi: [top10bahasabukuterpopuler.png](https://github.com/im-dheyy/Sentimen-Analytic/blob/main/Gambar/top10bahasabukuterpopuler.png)

### Top 10 Buku Berdasarkan Jumlah Rating

* Daftar buku yang paling sering dirating.
* Visualisasi: [top10bukudenganratingterbanyak.png](https://github.com/im-dheyy/Sentimen-Analytic/blob/main/Gambar/top10bukudenganratingterbanyak.png)

### Top 10 Kategori Buku

* Menunjukkan jenis kategori buku yang paling umum atau populer.
* Visualisasi: [top10kategoribuku.png](https://github.com/im-dheyy/Sentimen-Analytic/blob/main/Gambar/top10kategoribuku.png)

---

## 6. Content-Based Filtering

Sistem rekomendasi berbasis konten bekerja dengan menganalisis metadata buku (judul, penulis, publisher).

### Langkah-langkah:

* Menggabungkan kolom judul, penulis, dan penerbit menjadi satu string deskripsi.
* Mengubah teks menjadi vektor numerik dengan TF-IDF Vectorizer.
* Menghitung kemiripan antar buku dengan cosine similarity.
* Memberikan rekomendasi buku berdasarkan input judul buku yang mirip secara konten.

### Contoh Output:

Jika pengguna menyukai "The Da Vinci Code", maka sistem merekomendasikan buku seperti:

* Angels & Demons
* Deception Point
* Digital Fortress

---

## 7. Collaborative Filtering

### a. KNN (K-Nearest Neighbors)

* Dataset disiapkan dalam bentuk matriks user-item.
* Digunakan algoritma KNN dengan cosine similarity.
* Model menemukan buku yang mirip berdasarkan perilaku rating pengguna lain.

### b. Cosine Similarity

* Dibuat matriks pivot dari rating eksplisit.
* Hitung kemiripan antar item berdasarkan rating.
* Output rekomendasi berdasarkan buku yang diberi rating tinggi oleh pengguna.

### Contoh Output:

Jika pengguna memberi rating tinggi ke "Harry Potter and the Sorcerer's Stone", sistem akan merekomendasikan buku serupa yang juga disukai pengguna lain.

---

## 8. Kesimpulan

* Sistem rekomendasi yang dibangun mengombinasikan dua pendekatan utama: content-based dan collaborative filtering.
* Content-based cocok untuk pengguna baru atau buku baru (cold-start problem).
* Collaborative filtering menghasilkan rekomendasi yang lebih personal berdasarkan perilaku kolektif pengguna.
* Kualitas rekomendasi bergantung pada kualitas dan kelengkapan data.

Proyek ini menunjukkan bagaimana pendekatan machine learning dan NLP dapat digunakan untuk menyelesaikan permasalahan nyata dalam domain literasi dan perbukuan.

---

## 9. Referensi

* Dataset: [https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset)
* Scikit-learn documentation: [https://scikit-learn.org](https://scikit-learn.org)
* Medium Articles: berbagai referensi tentang sistem rekomendasi dengan Python
* Dokumentasi resmi Pandas, Matplotlib, Seaborn, dan Surprise Library
* Repositori Gambar & Kode: [https://github.com/im-dheyy/Sentimen-Analytic](https://github.com/im-dheyy/Sentimen-Analytic)
