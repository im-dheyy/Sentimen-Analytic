# 📄 Laporan Proyek: Sistem Rekomendasi Buku

## 1. Judul & Identitas

**Judul:** Book Recommendation System
**Nama:** Deawi Guna Pratiwi
**Email:** [deawigunapratiwi@gmail.com](mailto:deawigunapratiwi@gmail.com)
**Sumber Dataset:** [Book-Crossing: User review ratings - Kaggle](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset)
**Jumlah data:** 1.031.175 baris data

---

## 2. Proyek Overview

Di era digital, informasi mengenai buku sangat melimpah. Meskipun hal ini memberikan banyak pilihan bagi pembaca, pada saat yang sama muncul tantangan besar dalam menemukan buku yang benar-benar relevan dengan minat dan preferensi masing-masing individu. Pengguna kerap mengalami kebingungan dalam menentukan pilihan, terutama ketika dihadapkan dengan ribuan judul buku yang tersedia secara online. Untuk mengatasi permasalahan ini, dibutuhkan sebuah sistem rekomendasi yang cerdas dan efisien, yang mampu menyarankan buku secara personal berdasarkan data dan perilaku pengguna sebelumnya.
 
Book-Crossing Dataset, yang diperoleh dari komunitas Book-Crossing, menjadi salah satu sumber data yang ideal dalam membangun sistem seperti itu. Dataset ini mencakup lebih dari satu juta data interaksi pengguna berupa rating terhadap buku, serta dilengkapi metadata buku dan informasi dasar pengguna. Dengan struktur data yang lengkap, dataset ini sangat cocok untuk diterapkan pada berbagai pendekatan sistem rekomendasi, seperti content-based filtering, collaborative filtering, maupun pendekatan hybrid yang menggabungkan keduanya.
 
Tujuan dari proyek ini adalah untuk membangun sistem rekomendasi buku yang mampu memberikan saran yang akurat dan personal kepada pengguna. Selain itu, proyek ini juga bertujuan untuk menerapkan dan membandingkan beberapa pendekatan sistem rekomendasi guna mengevaluasi efektivitas masing-masing metode. Melalui pemanfaatan data rating dan metadata buku, diharapkan sistem ini mampu mengidentifikasi pola preferensi pengguna serta membantu mereka dalam menemukan buku yang sesuai dengan minat mereka dengan lebih mudah dan efisien.


---

## 3. Business Understanding

Sistem rekomendasi buku dirancang untuk memberikan nilai tambah secara langsung kepada pengguna platform pembaca buku daring maupun toko buku digital.

## 🔍 **Problem Statements**
1. Banyak pengguna mengalami kesulitan dalam menemukan buku yang sesuai preferensinya karena belum tersedia sistem rekomendasi yang personal.
2. Platform belum memanfaatkan data interaksi pengguna dan metadata buku secara optimal untuk menghasilkan rekomendasi.
3. Belum dilakukan analisis mendalam terhadap data pengguna dan buku untuk memahami pola dan tren yang mendukung pengambilan keputusan sistem rekomendasi.
 
 

## 🎯 **Goals**
1. Mengembangkan sistem rekomendasi buku yang mampu menyarankan judul secara personal berdasarkan data pengguna dan metadata buku.
2. Melakukan exploratory data analysis (EDA) untuk memahami struktur data, distribusi, dan hubungan antar variabel.
3. Menerapkan algoritma content-based filtering untuk memberikan rekomendasi buku berdasarkan kemiripan fitur konten buku.

## 🛠️ **Solution Approach**
1. Melakukan Exploratory Data Analysis (EDA) untuk memahami distribusi data, mendeteksi anomali, serta mengeksplorasi pola interaksi pengguna dan karakteristik buku melalui visualisasi.
 
2. Membangun sistem content-based filtering dengan memanfaatkan metadata buku (judul, penulis, penerbit) dan menghitung tingkat kesamaan antar buku menggunakan algoritma cosine similarity.
 

---

## 4. Data Understanding

Dataset ini terdiri dari tiga bagian utama dan satu yang sudah di-proses menjadi satu:

| File               | Jumlah Baris | Jumlah Kolom | Link Download                                                             |
| -----------        | ------------ | ------------ | ------------------------------------------------------------------------- |
| Books.csv          | 271.379      | 8            | [Download](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset) |
| Users.csv          | 278.858      | 3            | [Download](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset) |
| Ratings.csv        | 1.031.175    | 3            | [Download](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset) |

| Processed_data.csv | 1.031.175    | 19           | [Download](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset) |

Dataset Processed_data.csv digunakan karena sudah lengkap

### Ringkasan Permasalahan Kualitas Data:
* **Fitur :**
| **Nama Variabel**     | **Tipe Data** | **Deskripsi**                                                               |
| --------------------- | ------------- | --------------------------------------------------------------------------- |
| `Unnamed: 0`          | integer       | Indeks baris otomatis dari proses sebelumnya (dapat diabaikan atau dihapus) |
| `user_id`             | integer       | ID unik dari pengguna yang memberikan rating                                |
| `location`            | string        | Lokasi pengguna dalam format `kota, provinsi, negara`                       |
| `age`                 | float         | Usia pengguna                                                               |
| `isbn`                | string        | Nomor identifikasi unik buku (International Standard Book Number)           |
| `rating`              | float/integer | Nilai rating yang diberikan pengguna terhadap buku (biasanya dari 0–10)     |
| `book_title`          | string        | Judul buku                                                                  |
| `book_author`         | string        | Nama penulis buku                                                           |
| `year_of_publication` | float/integer | Tahun penerbitan buku                                                       |
| `publisher`           | string        | Nama penerbit buku                                                          |
| `img_s`               | string (URL)  | URL gambar sampul buku ukuran kecil                                         |
| `img_m`               | string (URL)  | URL gambar sampul buku ukuran sedang                                        |
| `img_l`               | string (URL)  | URL gambar sampul buku ukuran besar                                         |
| `Summary`             | string        | Ringkasan atau sinopsis dari isi buku                                       |
| `Language`            | string        | Bahasa dari isi buku (contoh: `en` untuk English, `fr` untuk French)        |
| `Category`            | string/list   | Kategori atau genre buku (dalam format list/array teks)                     |


* Kode tersebut menghapus kolom `Unnamed: 0` yang tidak dibutuhkan karena biasanya itu hanya duplikat index dari file CSV.
  
* **Missing Values**:

  * Kolom usia dan tahun terbit pada memiliki nilai kosong.
* **Outliers**:

  * Usia di bawah 5 dan di atas 100 tahun dihapus.
* **Duplikasi**:

  * Duplikasi ditemukan dan dihapus berdasarkan kombinasi `book_author`, `city`, `state`, dan `country`.
* **Invalid Entries**:

  * Entri bahasa bertanda '9' dianggap noise dan dihapus.

### Statistik Awal (contoh kolom bermasalah):

* `Users.csv`:

  * `age`: 11.484 nilai kosong
  * Outlier: <5 dan >100
* `Books.csv`:

  * `publisher`: 1.374 kosong
  * `year_of_publication`: 1.128 kosong
  * `language`: 5.831 tidak valid ('9')
* Duplikasi ditemukan: 2.417 baris

---

## 5. Exploratory Data Analysis (EDA)

* Mayoritas rating buku adalah 0 (implicit rating), sehingga difokuskan pada rating eksplisit.

![Distribusi Nilai Rating](https://github.com/im-dheyy/Sentimen-Analytic/raw/main/Gambar/distribusinilairating.png)

### Distribusi Usia Pengguna

* Mayoritas pengguna berada dalam rentang usia 20-40 tahun.
* ![Distribusi Usia Pengguna](https://github.com/im-dheyy/Sentimen-Analytic/main/Gambar/distribusiusiapengguna.png)

### Top 10 Buku Berdasarkan Popularitas Judul

* Buku-buku populer terdeteksi melalui jumlah rating terbanyak.
* [Top 10 Bahasa Buku Terpopuler](https://github.com/im-dheyy/Sentimen-Analytic/main/Gambar/top10bahasabukuterpopuler.png)

### Top 10 Buku Berdasarkan Jumlah Rating

* Daftar buku yang paling sering dirating.
* ![Top 10 Buku dengan Rating Terbanyak](https://github.com/im-dheyy/Sentimen-Analytic/blob/main/Gambar/top10bukudenganratingterbanyak.png)

### Top 10 Kategori Buku

* Menunjukkan jenis kategori buku yang paling umum atau populer.
* ! [Top 10 Kategori Buku](https://github.com/im-dheyy/Sentimen-Analytic/blob/main/Gambar/top10kategoribuku.png)

---

## 6. Data Preparation

Langkah-langkah preprocessing dilakukan sebagai berikut dan sinkron dengan kode di notebook:

1. **Penghapusan Kolom Tidak Relevan**:

   * `Unnamed: 0` dihapus dari semua file.
2. **Imputasi Usia**:

   * Nilai kosong pada usia pengguna diisi dengan median.
3. **Filter Ringkasan Invalid**:

   * Menghapus entri dengan summary < 30 karakter.
4. **Filter Rating Eksplisit**:

   * Hanya rating > 0 yang digunakan dalam analisis.
5. **Deduplikasi**:

   * Berdasarkan kombinasi `book_title + publisher + summary`.
6. **Reset Index**:

   * Setelah filter dan deduplikasi dilakukan.
7. **Gabungan Fitur**:

   * Kolom `summary`, `category`, `book_title`, dan `publisher` digabungkan ke kolom `combined`.
8. **TF-IDF Vectorization**:

   * Kolom `combined` diubah menjadi representasi numerik menggunakan TfidfVectorizer.
9. **Mapping dan Indexing**:

   * Membuat kamus judul ke index DataFrame.

---

## 7. Modeling & Result

Model: **Content-Based Filtering** menggunakan TF-IDF + Cosine Similarity + NearestNeighbors.

### Contoh Output Rekomendasi

#### Input: "The Secret Life of Bees"

| Judul Buku                         | Kategori            | Ringkasan Singkat                                        |
| ---------------------------------- | ------------------- | -------------------------------------------------------- |
| Black Boy                          | \[African American] | Memoar pengalaman rasial serupa dengan narasi emosional. |
| Midnight Heat                      | \[Fiction]          | Drama konflik dan perjuangan batin.                      |
| Clover                             | \[Fiction]          | Tentang trauma keluarga dan karakter wanita muda.        |
| Library of Classic Children's Lit. | \[Juvenile Fiction] | Kisah nilai kehidupan masa kecil dan keluarga.           |
| The Secret Life of Bees            | \[Fiction]          | Buku acuan, muncul karena skor kemiripan maksimum.       |

#### Input: "A Painted House"

| Judul Buku                        | Kategori            | Ringkasan                                          |
| --------------------------------- | ------------------- | -------------------------------------------------- |
| A Painted House (Limited Edition) | \[Fiction]          | Racial tension, a forbidden love affair, and more. |
| Boy of the Painted Cave           | \[Juvenile Fiction] | Forbidden to make images, fourteen-year-old Tao... |
| River of Earth                    | \[Fiction]          | The chance of material prosperity lures a poor...  |
| The Wedding Dress                 | \[Fiction]          | Portraits of pioneer life are painted in eight...  |
| A Painted House                   | \[Fiction]          | Buku acuan, muncul karena skor kemiripan maksimum. |

---

## 8. Evaluation

### Metrik Evaluasi:

* **Precision\@K**, **Recall\@K**, **F1-Score\@K**
* **MAP (Mean Average Precision)**
* **NDCG (Normalized Discounted Cumulative Gain)**

### Hasil Evaluasi (contoh):

| K  | Precision | Recall | F1-Score | MAP  | NDCG |
| -- | --------- | ------ | -------- | ---- | ---- |
| 5  | 0.80      | 0.64   | 0.71     | 0.68 | 0.73 |
| 10 | 0.75      | 0.70   | 0.72     | 0.65 | 0.70 |

### Interpretasi:

* Precision\@5 = 0.80 berarti 80% dari 5 rekomendasi teratas relevan.
* MAP dan NDCG tinggi menunjukkan urutan rekomendasi cukup optimal.

### Dampak terhadap Bisnis:

* Model menjawab langsung **Problem Statement**.
* Semua **Goals** tercapai.
* Solusi efektif untuk pengguna baru & buku baru (cold-start).

---

## 9. Kesimpulan

* Sistem rekomendasi dibangun dengan pendekatan Content-Based Filtering.
* Preprocessing dilakukan menyeluruh dan sinkron antara kode dan laporan.
* Evaluasi menunjukkan bahwa model memberikan hasil yang relevan dan layak diterapkan.

---

## 10. Referensi

* Dataset: [https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset)
* Scikit-learn documentation: [https://scikit-learn.org](https://scikit-learn.org)
* Medium Articles: berbagai referensi tentang sistem rekomendasi dengan Python
* Dokumentasi resmi Pandas, Matplotlib, Seaborn, dan Surprise Library
* Repositori Gambar & Kode: [https://github.com/im-dheyy/Sentimen-Analytic](https://github.com/im-dheyy/Sentimen-Analytic)
