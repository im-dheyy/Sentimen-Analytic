## 📌 Laporan Proyek: Sistem Rekomendasi Buku

### Project Overview

Sistem rekomendasi buku berbasis konten (content-based filtering) dikembangkan untuk membantu pengguna menemukan buku-buku yang mirip dengan preferensi mereka berdasarkan deskripsi buku, kategori, judul, dan penerbit. Sistem ini tidak bergantung pada rating pengguna lain, sehingga cocok digunakan untuk kondisi cold-start atau ketika data rating sangat terbatas.

### Business Understanding

Permasalahan yang ingin diselesaikan adalah bagaimana menyarankan buku yang relevan kepada pengguna meskipun tidak memiliki histori rating pengguna lain. Tujuannya adalah meningkatkan pengalaman pengguna dalam menjelajahi koleksi buku, serta mendorong eksplorasi buku-buku serupa yang mungkin belum pernah dibaca sebelumnya.

### Data Understanding

Dataset yang digunakan merupakan hasil penggabungan dari beberapa atribut buku seperti:

* Judul buku (`book_title`)
* Ringkasan buku (`Summary`)
* Kategori buku (`Category`)
* Nama penerbit (`publisher`)

Data ini memiliki lebih dari 1 juta entri yang mencakup informasi pengguna, lokasi, buku, dan metadata lainnya. Fokus utama sistem rekomendasi berada pada konten dari masing-masing buku.

### Data Preparation

1. **Pembersihan Data**:

   * Menghapus entri dengan ringkasan tidak valid (misal hanya "9")
   * Menghapus entri dengan kategori kosong (`[]`)
   * Menghapus ringkasan dengan panjang kurang dari 30 karakter
   * Menghapus duplikasi berdasarkan kombinasi `book_title`, `publisher`, dan `Summary`

2. **Penggabungan Fitur**:

   * Membuat kolom `combined` yang terdiri dari gabungan `Summary`, `Category`, `book_title`, dan `publisher`

3. **Transformasi TF-IDF**:

   * Data dalam kolom `combined` diubah menjadi representasi numerik menggunakan TF-IDF Vectorizer.

4. **Pemetaan Judul**:

   * Dibuat pemetaan antara judul buku dan indeks baris untuk akses cepat saat melakukan rekomendasi.

### Modelling and Result

* Model `NearestNeighbors` dari `sklearn` digunakan untuk menghitung kemiripan antar buku berdasarkan cosine similarity dari vektor TF-IDF.
* Sistem menerima input berupa judul buku, lalu mencari 5 buku paling mirip (selain dirinya sendiri) berdasarkan representasi kontennya.
* Contoh hasil rekomendasi menunjukkan bahwa sistem mampu menyarankan buku-buku yang memiliki tema, kategori, dan gaya narasi yang serupa.

### Evaluation

* Evaluasi dilakukan dengan cara observasi terhadap hasil rekomendasi dari beberapa input judul buku.
* Sistem berhasil menyarankan judul lain dengan genre dan tema yang konsisten.
* Sistem terbukti efektif menangani kasus cold-start, karena tidak membutuhkan data rating dari pengguna lain.

### Struktur Laporan

1. **Project Overview**: Deskripsi umum proyek dan tujuannya
2. **Business Understanding**: Latar belakang bisnis dan objektif sistem rekomendasi
3. **Data Understanding**: Informasi mengenai struktur dan isi data yang digunakan
4. **Data Preparation**: Langkah-langkah pra-pemrosesan dan pembersihan data
5. **Modelling and Result**: Penjelasan algoritma yang digunakan dan hasil yang diperoleh
6. **Evaluation**: Penilaian hasil sistem dan validitas rekomendasi
7. **Kesimpulan dan Saran** (opsional): Insight akhir dari proyek dan saran pengembangan lanjut
