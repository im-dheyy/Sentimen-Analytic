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

Sistem rekomendasi buku dirancang untuk memberikan nilai tambah secara langsung kepada pengguna platform pembaca buku daring maupun toko buku digital.

### Problem Statement:

* Pengguna kesulitan menemukan buku yang sesuai dengan minat mereka di tengah banyaknya pilihan.
* Tidak adanya sistem rekomendasi yang mempertimbangkan konten dan preferensi eksplisit pengguna.

### Goals:

* Mengembangkan sistem yang dapat memberikan rekomendasi buku yang relevan berdasarkan preferensi pengguna.
* Menyediakan solusi yang dapat diimplementasikan untuk cold-start user dan buku baru.
* Meningkatkan keterlibatan pengguna melalui personalisasi.

### Solution Statement:

* Mengimplementasikan pendekatan Content-Based Filtering berbasis konten buku seperti ringkasan, judul, kategori, dan penerbit.
* Menyediakan output rekomendasi yang dapat divisualisasikan dan dievaluasi menggunakan metrik yang sesuai seperti Precision\@K dan Recall\@K. untuk mencakup berbagai skenario pengguna.
* Menyediakan output rekomendasi yang dapat divisualisasikan dan dievaluasi menggunakan metrik yang sesuai seperti Precision\@K dan Recall\@K.

---

## 4. Data Understanding

Dataset ini terdiri dari tiga bagian utama:

* **Books.csv**: Memuat 271.379 entri buku dengan fitur seperti ISBN, judul, penulis, penerbit, tahun terbit, dan kategori.
* **Users.csv**: Menyediakan 278.858 entri pengguna, termasuk ID, lokasi, dan usia.
* **Ratings.csv**: Terdiri dari 1.031.175 baris, yang mencatat rating pengguna terhadap buku (rentang 0–10).

### Cakupan & Kualitas Data:

* Jumlah fitur: 16 kolom
* Jumlah data: 1.031.175 baris

### Permasalahan Kualitas Data:

* **Missing Values**:

  * Beberapa entri pada usia pengguna dan tahun terbit tidak tersedia.
* **Outliers**:

  * Usia di bawah 5 tahun dan di atas 100 tahun dianggap sebagai outlier dan dihapus.
* **Duplikasi**:

  * Ditemukan pada judul dan ISBN yang serupa, dilakukan deduplikasi berdasarkan kombinasi `book_title + publisher + summary`.
* **Invalid Entries**:

  * Bahasa bertanda '9' diidentifikasi sebagai noise dan dihapus.

### Aksi Pembersihan:

* Menghapus fitur `Unnamed: 0`.
* Menghapus nilai kosong dan duplikasi.
* Imputasi nilai usia dengan median.
* Memfilter hanya rating eksplisit (rating > 0).

---

## 5. Exploratory Data Analysis (EDA)

EDA dilakukan untuk memahami karakteristik data, mendeteksi pola, dan anomali. Visualisasi dibuat untuk memberikan insight yang lebih mendalam terhadap distribusi data.

### Distribusi Rating Buku

Berdasarkan grafik distribusi nilai rating:

* Rating 0 mendominasi jumlah data secara signifikan, dengan lebih dari 600.000 entri. Ini kemungkinan besar menunjukkan rating implisit atau tidak ada rating yang diberikan oleh pengguna.
* Rating dari 1 hingga 10 jumlahnya jauh lebih sedikit, dengan puncak pada rating 8, diikuti oleh rating 7, 10, dan 5.
* Distribusi rating menunjukkan bahwa ketika pengguna memberikan rating eksplisit, mereka cenderung memberikan nilai tinggi (positif).

**Kesimpulan:**
Mayoritas data berisi rating 0 (bisa dianggap sebagai "tidak diketahui" atau "tidak diberikan"), dan data rating eksplisit menunjukkan kecenderungan bias positif. Hal ini perlu dipertimbangkan dalam pemodelan, misalnya dengan memisahkan rating 0 dari rating eksplisit.

![Distribusi Nilai Rating](https://github.com/im-dheyy/Sentimen-Analytic/raw/main/Gambar/distribusinilairating.png)

### Distribusi Usia Pengguna

Berdasarkan grafik Distribusi Usia Pengguna:

* Mayoritas pengguna berada pada rentang usia 30–40 tahun, dengan puncak tertinggi sekitar usia 35 tahun.
* Terlihat adanya penurunan tajam di luar rentang usia tersebut, terutama setelah usia 60 tahun.
* Distribusi menunjukkan pola right-skewed, artinya lebih banyak pengguna muda hingga paruh baya dibandingkan pengguna lansia.

**Kesimpulan:**
Sistem rekomendasi buku sebaiknya mempertimbangkan dominasi usia 30–40 tahun ini, karena preferensi mereka kemungkinan besar paling merepresentasikan tren umum dalam data.

![Distribusi Usia Pengguna](https://github.com/im-dheyy/Sentimen-Analytic/raw/main/Gambar/distribusiusiapengguna.png)

### Top 10 Bahasa Buku Terpopuler

Grafik Top 10 Bahasa Buku Terpopuler menunjukkan distribusi bahasa dari buku-buku yang tersedia dalam dataset:

* Bahasa Inggris (en) mendominasi secara signifikan dengan jumlah buku terbanyak (lebih dari 600.000 judul).
* Label ‘9’ muncul sebagai bahasa kedua terbanyak, yang kemungkinan besar merupakan data anomali atau kode yang salah dalam pengisian kolom bahasa.
* Bahasa lainnya seperti Jerman (de), Spanyol (es), Prancis (fr), dan beberapa bahasa Eropa lainnya muncul dalam jumlah yang jauh lebih kecil.

**Kesimpulan:**
Mayoritas buku dalam dataset berbahasa Inggris, sehingga sistem rekomendasi kemungkinan akan lebih relevan bagi pengguna yang memahami bahasa tersebut. Perlu penanganan khusus terhadap entri bahasa tidak valid seperti ‘9’.

![Top 10 Judul Buku Terpopuler](https://github.com/im-dheyy/Sentimen-Analytic/raw/main/Gambar/top10bahasabukuterpopuler.png)

### Top 10 Buku Berdasarkan Jumlah Rating

Berdasarkan grafik Top 10 Buku dengan Rating Terbanyak:

* Buku "Wild Animus" memperoleh jumlah rating terbanyak secara signifikan dibandingkan buku lain, melebihi 2.500 rating.
* Buku populer lain seperti "The Lovely Bones: A Novel", "The Da Vinci Code", dan "The Secret Life of Bees" juga masuk dalam daftar dengan jumlah rating yang tinggi.
* Grafik ini mencerminkan buku-buku yang paling sering dinilai oleh pengguna, bukan buku dengan rating tertinggi secara kualitas.

**Kesimpulan:**
Buku-buku pada grafik ini bisa dianggap sebagai buku populer atau paling dikenal oleh komunitas pengguna, sehingga cocok dijadikan acuan awal untuk sistem rekomendasi berbasis popularitas atau cold-start (tanpa data pengguna).

![Top 10 Buku dengan Rating Terbanyak](https://github.com/im-dheyy/Sentimen-Analytic/raw/main/Gambar/top10bukudenganratingterbanyak.png)

### Top 10 Kategori Buku

Grafik Top 10 Kategori Buku menunjukkan distribusi kategori buku yang paling banyak terdapat dalam dataset:

* Kategori "Fiction" mendominasi secara signifikan, dengan jumlah buku hampir 400.000 judul. Ini menunjukkan bahwa fiksi adalah genre yang paling umum dan populer dalam koleksi data ini.
* Diikuti oleh "Juvenile Fiction" (fiksi untuk remaja) dan "Biography & Autobiography", yang juga memiliki jumlah signifikan namun jauh lebih kecil dibandingkan fiksi umum.
* Kategori lain seperti "Humor", "History", dan "Religion" muncul dalam jumlah lebih terbatas.

**Kesimpulan:**
Genre fiksi mendominasi isi dataset, sehingga sistem rekomendasi kemungkinan akan lebih banyak merekomendasikan buku-buku dalam kategori ini kecuali dilakukan penyesuaian khusus untuk genre lainnya.

![Top 10 Kategori Buku](https://github.com/im-dheyy/Sentimen-Analytic/raw/main/Gambar/top10kategoribuku.png)


---


## 6. Data Preparation

Tahapan data preparation meliputi:

* **Dropping Kolom Tidak Relevan**: Kolom seperti `Unnamed: 0` dihapus.
* **Imputasi Usia**: Menggunakan nilai median untuk menggantikan nilai kosong.
* **Filter Ringkasan**: Menghapus ringkasan dengan panjang < 30 karakter.
* **Gabungan Fitur**: Menggabungkan `summary`, `book_title`, `category`, `publisher` menjadi satu kolom `combined`.
* **Deduplication**: Penghapusan duplikat berdasarkan kolom `combined`.
* **TF-IDF Vectorization**: Representasi teks ke dalam vektor numerik.
* **Mapping dan Indexing**: Pemetaan judul ke index untuk efisiensi pencarian.

---

## 7. Modeling & Result

Model yang digunakan:

* **Content-Based Filtering**: Menggunakan TF-IDF dan NearestNeighbors (cosine similarity).
* **Output**:

  * Untuk input *"The Secret Life of Bees"*, sistem memberikan 5 rekomendasi dengan kemiripan tinggi secara konten.
* **Top-N Recommendation**:

  * Disediakan dalam bentuk tabel hasil pemanggilan fungsi `recommend_books_nn()`.

---

## 8. Evaluation

Evaluasi dilakukan dengan pendekatan berikut:

* **Precision\@K** dan **Recall\@K**: Untuk menilai proporsi dan jangkauan relevansi rekomendasi.
* **F1-Score\@K**: Kombinasi dari Precision dan Recall.
* **MAP & NDCG**: Digunakan untuk mengukur kualitas ranking dan distribusi relevansi rekomendasi.

### Skema Evaluasi:

* Semua perhitungan dilakukan langsung melalui notebook.
* Output model dibandingkan pada berbagai input judul untuk mengevaluasi konsistensi dan relevansi hasil.

### Business Impact:

* Sistem telah menjawab **problem statement** secara langsung.
* Semua **goals** telah tercapai dengan keberhasilan model memberikan hasil Top-N rekomendasi yang relevan.
* Dapat diterapkan pada skenario **cold-start** dan personalisasi berbasis konten.

---

## 9. Kesimpulan

* Sistem rekomendasi dibangun dengan pendekatan Content-Based Filtering.
* Telah dilakukan pembersihan, transformasi, dan deduplikasi data secara sistematis.
* Output model diuji dan menghasilkan rekomendasi yang relevan secara tematik dan konten.
* Evaluasi dilakukan dengan metrik rekomendasi modern.

---

## 10. Referensi

* Dataset: [https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset)
* Scikit-learn documentation: [https://scikit-learn.org](https://scikit-learn.org)
* Medium Articles: berbagai referensi tentang sistem rekomendasi dengan Python
* Dokumentasi resmi Pandas, Matplotlib, Seaborn, dan Surprise Library
* Repositori Gambar & Kode: [https://github.com/im-dheyy/Sentimen-Analytic](https://github.com/im-dheyy/Sentimen-Analytic)


---

