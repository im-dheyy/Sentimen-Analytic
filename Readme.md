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

## 6. Content-Based Filtering

Dalam pendekatan Content-Based Filtering, sistem rekomendasi akan memberikan saran buku yang mirip secara konten dengan buku yang pernah disukai pengguna. Fokus utama metode ini adalah menganalisis informasi deskriptif dari setiap item (dalam hal ini, buku) untuk menemukan kemiripan antar buku berdasarkan atribut-atribut berikut:

📖 Summary: Ringkasan isi buku yang menggambarkan topik atau cerita.

🏷️ Category: Kategori atau genre buku, misalnya [Fiction], [Science], [Biography], dll.

📘 book_title: Judul buku, karena bisa mengandung kata-kata penting terkait isi.

🏢 publisher: Nama penerbit, yang dalam beberapa kasus mengindikasikan genre atau kualitas buku.



### Langkah-langkah:

 1. Pembersihan dan Filter Data
 - Menghapus entri dengan ringkasan tidak valid ('9')
 - Menghapus entri dengan kategori kosong ([])
 - Menghapus ringkasan yang terlalu pendek (< 30 karakter)
 - Menghapus duplikasi berdasarkan book_title + publisher + Summary

 2. Penggabungan Fitur Teks
 - Menggabungkan kolom Summary, Category, book_title, dan publisher ke kolom 'combined'
 - Ini bertujuan menyatukan informasi penting dalam satu representasi teks yang kaya konten

 3. Penghapusan Duplikasi dan Reset Index
 - Membersihkan data dari duplikasi berdasarkan kolom 'combined'
 - Reset index untuk memastikan baris DataFrame sejajar dengan tfidf_matrix

 4. TF-IDF Vectorization
 - Mengubah data teks pada kolom 'combined' menjadi vektor numerik dengan TfidfVectorizer
 - TF-IDF membantu menekankan kata-kata penting dan menurunkan bobot kata umum

 5. Pelatihan Model Nearest Neighbors
 - Melatih model NearestNeighbors dengan tfidf_matrix
 - Menggunakan cosine similarity untuk mengukur kemiripan antar buku

 6. Pemetaan Judul Buku ke Index
 - Membuat mapping dari judul buku (dalam lowercase) ke index DataFrame
 - Memungkinkan akses langsung ke representasi vektor berdasarkan input judul buku

 📌 Hasil Akhir
 Sistem rekomendasi content-based siap digunakan untuk menyarankan buku yang mirip
 berdasarkan isi dan metadata, tanpa bergantung pada rating pengguna lain.
 Sangat cocok untuk cold-start scenario seperti buku baru yang belum memiliki rating.

## Model and Result
Menemukan buku-buku yang paling mirip dengan buku input, sehingga bisa memberikan rekomendasi yang relevan secara konten, meskipun belum pernah diberi rating oleh pengguna.

Langkah-Langkah Penjelasan Fungsi:
1. Fungsi menerima judul buku yang ingin dicari kemiripannya.
2. Menggunakan TF-IDF untuk mengukur kemiripan konten antar buku.
3. Model NearestNeighbors digunakan untuk mencari buku dengan vektor TF-IDF paling dekat (mirip).
4. Mengembalikan beberapa buku yang memiliki konten mirip berdasarkan ringkasan, kategori, judul, dan penerbit.

### Penjelasan Output Rekomendasi: "The Secret Life of Bees"

Hasil dari pemanggilan fungsi `recommend_books_nn("The Secret Life of Bees", top_n=5)` memberikan 5 buku yang direkomendasikan berdasarkan kemiripan konten dengan buku input. Rekomendasi ini dihasilkan dari model content-based filtering yang membandingkan fitur teks gabungan (summary, kategori, judul, dan penerbit).

Berikut adalah penjelasan untuk masing-masing buku hasil rekomendasi:

1. **Black Boy**
   - **Kategori**: [African American authors]
   - **Alasan Rekomendasi**: Buku ini memiliki konten naratif yang kuat tentang pengalaman hidup, mirip dengan tema pencarian identitas dalam *The Secret Life of Bees*.

2. **Midnight Heat**
   - **Kategori**: [Fiction]
   - **Alasan Rekomendasi**: Novel fiksi dengan elemen emosi kuat dan konflik sosial, cocok dengan pembaca yang menyukai dinamika karakter dan perjuangan batin.

3. **Clover**
   - **Kategori**: [Fiction]
   - **Alasan Rekomendasi**: Sama-sama menceritakan hubungan keluarga dan trauma kehilangan, dengan karakter wanita muda sebagai tokoh utama.

4. **LIBRARY OF CLASSIC CHILDREN'S LITERATURE**
   - **Kategori**: [Juvenile Fiction]
   - **Alasan Rekomendasi**: Meskipun ditujukan untuk anak-anak, rekomendasi muncul karena kemiripan gaya naratif dan nilai-nilai emosional yang terkandung dalam cerita.

5. **The Secret Life of Bees**
   - **Kategori**: [Fiction]
   - **Catatan**: Buku ini sendiri tetap muncul sebagai hasil teratas, karena secara teknis memiliki kemiripan tertinggi dengan dirinya sendiri. Namun hasil akhir hanya menampilkan buku-buku selain buku input, sesuai logika dalam fungsi.

### Kesimpulan
Rekomendasi yang dihasilkan menunjukkan bahwa sistem dapat mengenali pola naratif, tema emosional, serta atribut tekstual lainnya yang relevan. Hal ini membuktikan bahwa pendekatan content-based cukup efektif meskipun tidak menggunakan data rating pengguna.

---
### 📚 Recommendation Output

#### Judul: *A Painted House*

Berdasarkan pendekatan content-based filtering, sistem memberikan rekomendasi buku yang memiliki kemiripan konten dengan *A Painted House* melalui analisis terhadap *summary*, *category*, *judul*, dan *publisher*. Berikut penjelasan hasil rekomendasinya:

1. **A Painted House (Delta)** — Buku utama dan referensi pencarian. Kisahnya mengangkat tema ketegangan rasial dan cinta terlarang, dengan latar kehidupan petani.

2. **A Painted House (Limited Edition) - Doubleday** — Edisi berbeda dari buku yang sama, tetap relevan karena memiliki isi konten dan cerita serupa.

3. **Boy of the Painted Cave** — Memiliki elemen tematik serupa seperti pembatasan sosial dan perjuangan hidup di masyarakat tertutup, ditulis dalam genre *Juvenile Fiction*.

4. **River of Earth** — Cerita tentang keluarga miskin yang berjuang untuk kehidupan lebih baik, serupa dengan latar ekonomi dan sosial di *A Painted House*.

5. **The Wedding Dress** — Menceritakan potret kehidupan perintis dengan gaya naratif yang mirip, menggambarkan masa lalu dan perjuangan hidup di era tertentu.

#### Kesimpulan:
Rekomendasi ini menunjukkan bahwa sistem berhasil mengidentifikasi buku-buku dengan kemiripan tema, genre, dan latar suasana, bahkan meskipun judul atau pengarang berbeda. Ini menegaskan keefektifan pendekatan content-based dalam menemukan buku sejenis berdasarkan isi konten, bukan hanya popularitas atau rating pengguna lain.

---
---

## 8. Kesimpulan

* Sistem rekomendasi yang dibangun satu pendekatan utama: content-based
* Content-based cocok untuk pengguna baru atau buku baru (cold-start problem).
* Kualitas rekomendasi bergantung pada kualitas dan kelengkapan data.

Proyek ini menunjukkan bagaimana pendekatan machine learning dan NLP dapat digunakan untuk menyelesaikan permasalahan nyata dalam domain literasi dan perbukuan.

---

## 9. Referensi

* Dataset: [https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset)
* Scikit-learn documentation: [https://scikit-learn.org](https://scikit-learn.org)
* Medium Articles: berbagai referensi tentang sistem rekomendasi dengan Python
* Dokumentasi resmi Pandas, Matplotlib, Seaborn, dan Surprise Library
* Repositori Gambar & Kode: [https://github.com/im-dheyy/Sentimen-Analytic](https://github.com/im-dheyy/Sentimen-Analytic)
