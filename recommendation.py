#!/usr/bin/env python
# coding: utf-8

# # **SYSTEM RECOMMENDATION : BOOK RECOMMENDATION SYSTEM**
# ## Nama : Deawi Guna Pratiwi
# ## Email : deawigunapratiwi@gmail.com
# ## Sumber Dataset :
# Dataset diperoleh dari kaggle dengan judul **Book-Crossing: User review ratings** (https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset) dengan jumlah dataset 1031175 data.

# # **Proyek Overview**

# Di era digital, informasi mengenai buku sangat melimpah. Meskipun hal ini memberikan banyak pilihan bagi pembaca, pada saat yang sama muncul tantangan besar dalam menemukan buku yang benar-benar relevan dengan minat dan preferensi masing-masing individu. Pengguna kerap mengalami kebingungan dalam menentukan pilihan, terutama ketika dihadapkan dengan ribuan judul buku yang tersedia secara online. Untuk mengatasi permasalahan ini, dibutuhkan sebuah sistem rekomendasi yang cerdas dan efisien, yang mampu menyarankan buku secara personal berdasarkan data dan perilaku pengguna sebelumnya.
# 
# Book-Crossing Dataset, yang diperoleh dari komunitas Book-Crossing, menjadi salah satu sumber data yang ideal dalam membangun sistem seperti itu. Dataset ini mencakup lebih dari satu juta data interaksi pengguna berupa rating terhadap buku, serta dilengkapi metadata buku dan informasi dasar pengguna. Dengan struktur data yang lengkap, dataset ini sangat cocok untuk diterapkan pada berbagai pendekatan sistem rekomendasi, seperti content-based filtering, collaborative filtering, maupun pendekatan hybrid yang menggabungkan keduanya.
# 
# Tujuan dari proyek ini adalah untuk membangun sistem rekomendasi buku yang mampu memberikan saran yang akurat dan personal kepada pengguna. Selain itu, proyek ini juga bertujuan untuk menerapkan dan membandingkan beberapa pendekatan sistem rekomendasi guna mengevaluasi efektivitas masing-masing metode. Melalui pemanfaatan data rating dan metadata buku, diharapkan sistem ini mampu mengidentifikasi pola preferensi pengguna serta membantu mereka dalam menemukan buku yang sesuai dengan minat mereka dengan lebih mudah dan efisien.

# # 💼 **Business Understanding**

# ## 🔍 **Problem Statements**
# 1. Banyak pengguna mengalami kesulitan dalam menemukan buku yang sesuai preferensinya karena belum tersedia sistem rekomendasi yang personal.
# 2. Platform belum memanfaatkan data interaksi pengguna dan metadata buku secara optimal untuk menghasilkan rekomendasi.
# 3. Belum dilakukan analisis mendalam terhadap data pengguna dan buku untuk memahami pola dan tren yang mendukung pengambilan keputusan sistem rekomendasi.
# 
# 

# ## 🎯 **Goals**
# 1. Mengembangkan sistem rekomendasi buku yang mampu menyarankan judul secara personal berdasarkan data pengguna dan metadata buku.
# 2. Melakukan exploratory data analysis (EDA) untuk memahami struktur data, distribusi, dan hubungan antar variabel.
# 3. Menerapkan algoritma content-based filtering untuk memberikan rekomendasi buku berdasarkan kemiripan fitur konten buku.

# ## 🛠️ **Solution Approach**
# 1. Melakukan Exploratory Data Analysis (EDA) untuk memahami distribusi data, mendeteksi anomali, serta mengeksplorasi pola interaksi pengguna dan karakteristik buku melalui visualisasi.
# 
# 2. Membangun sistem content-based filtering dengan memanfaatkan metadata buku (judul, penulis, penerbit) dan menghitung tingkat kesamaan antar buku menggunakan algoritma cosine similarity.
# 
# 

# # **Data Understanding**

# ## **Import Library**
# Import semua library yang diperlukan

# In[1]:


import os
import shutil
import zipfile
import re
from IPython.display import display
import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report


# In[2]:


#membuka zip menjadi folder
# Path file zip yang kamu upload
zip_path = "archive.zip"
extract_to = "/bookcrossing_data"

# Mengekstrak jika folder belum ada
if not os.path.exists(extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Berhasil mengekstrak ke: {extract_to}")
else:
    print("📁 Folder sudah ada, ekstraksi dilewati.")


# In[5]:


# Menelusuri semua file dan subfolder
for root, dirs, files in os.walk(extract_to):
    print(f"\n📁 Folder: {root}")
    for f in files:
        print("  -", f)


# In[7]:


# load file dari folder Books Data with Category Language and Summary
preprocessed = pd.read_csv(f"{extract_to}/Books Data with Category Language and Summary/Preprocessed_data.csv")


# In[8]:


# Display the first few rows
preprocessed.head()


# ## **Deskripsi Variabel**

# | **Nama Variabel**     | **Tipe Data** | **Deskripsi**                                                               |
# | --------------------- | ------------- | --------------------------------------------------------------------------- |
# | `Unnamed: 0`          | integer       | Indeks baris otomatis dari proses sebelumnya (dapat diabaikan atau dihapus) |
# | `user_id`             | integer       | ID unik dari pengguna yang memberikan rating                                |
# | `location`            | string        | Lokasi pengguna dalam format `kota, provinsi, negara`                       |
# | `age`                 | float         | Usia pengguna                                                               |
# | `isbn`                | string        | Nomor identifikasi unik buku (International Standard Book Number)           |
# | `rating`              | float/integer | Nilai rating yang diberikan pengguna terhadap buku (biasanya dari 0–10)     |
# | `book_title`          | string        | Judul buku                                                                  |
# | `book_author`         | string        | Nama penulis buku                                                           |
# | `year_of_publication` | float/integer | Tahun penerbitan buku                                                       |
# | `publisher`           | string        | Nama penerbit buku                                                          |
# | `img_s`               | string (URL)  | URL gambar sampul buku ukuran kecil                                         |
# | `img_m`               | string (URL)  | URL gambar sampul buku ukuran sedang                                        |
# | `img_l`               | string (URL)  | URL gambar sampul buku ukuran besar                                         |
# | `Summary`             | string        | Ringkasan atau sinopsis dari isi buku                                       |
# | `Language`            | string        | Bahasa dari isi buku (contoh: `en` untuk English, `fr` untuk French)        |
# | `Category`            | string/list   | Kategori atau genre buku (dalam format list/array teks)                     |
# 

# In[11]:


preprocessed.info()


# In[12]:


# Menampilkan jumlah baris dan kolom pada data
total_row, total_column = preprocessed.shape
print(f"Total of rows: {total_row}")
print(f"Total of column: {total_column}")


# Dapat dilihat bahwa data yang digunakan adalah sebanyak 1031175 data dengan 19 fitur dengan terdapat 2 variabel bertipe `float64`, 3 variabel bertipe `int64`, 14 variabel bertipe `object`.

# ## Statistik Deskripsi dari Data

# In[15]:


preprocessed.describe()


# Tabel di atas memberikan informasi statistik pada masing-masing kolom, antara lain:
# - Count adalah jumlah sampel pada data.
# - Mean adalah nilai rata-rata.
# - Std adalah standar deviasi (mengukur seberapa tersebar data).
# - Min yaitu nilai minimum setiap kolom.
# - 25% adalah kuartil pertama, yaitu nilai di bawah 25% data berada.
# - 50% adalah kuartil kedua, juga disebut median (nilai tengah data).
# - 75% adalah kuartil ketiga, yaitu nilai di bawah 75% data berada.
# - Max adalah nilai maksimum

# Penjelasan:
# 
# | **Kolom**             | **Penjelasan Statistik**                                                                                                                                                                                                                                                            |
# | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | `Unnamed: 0`          | Ini adalah indeks baris otomatis yang di-generate dari proses sebelumnya. Nilainya unik dan tidak penting untuk analisis.                                                                                                                                                           |
# | `user_id`             | ID pengguna bersifat numerik. Nilai `min = 2` dan `max = 278854`, menunjukkan jumlah pengguna aktif yang cukup besar. Tidak ada indikasi kejanggalan di sini.                                                                                                                       |
# | `age`                 | Usia pengguna berkisar antara **5 hingga 99 tahun**. Rata-rata usia pengguna adalah sekitar **36 tahun**, dengan standar deviasi sekitar **10 tahun**, artinya mayoritas pengguna berada dalam rentang **25–45 tahun**.                                                             |
# | `rating`              | Skor rating dari pengguna terhadap buku berada dalam skala **0 hingga 10**. Nilai **mean-nya hanya 2.83**, dan nilai median **0**, menunjukkan bahwa mayoritas pengguna tidak memberikan rating aktif (0 = no rating). Hanya sebagian kecil yang memberi nilai tinggi seperti 7–10. |
# | `year_of_publication` | Tahun penerbitan buku berkisar dari **1376 hingga 2008**. Nilai minimum (1376) sangat mungkin merupakan outlier (data salah input). Mayoritas buku diterbitkan di sekitar **1992–2001**, dan tahun median-nya adalah **1997**.                                                      |
# 

# In[18]:


preprocessed_clean = preprocessed.drop(columns=['Unnamed: 0'])


# Kode tersebut menghapus kolom `Unnamed: 0` yang tidak dibutuhkan karena biasanya itu hanya duplikat index dari file CSV.

# In[20]:


preprocessed_clean = preprocessed_clean[preprocessed_clean['age'].between(10, 90)]


# kode tersebut menyaring data agar hanya menyertakan pengguna dengan usia antara 10 hingga 90 tahun.

# In[22]:


# Ubah ke integer terlebih dahulu (jika belum)
preprocessed_clean['year_of_publication'] = preprocessed_clean['year_of_publication'].astype(float)

# Ganti outlier dengan NaN
preprocessed_clean.loc[
    (preprocessed_clean['year_of_publication'] < 1500) | 
    (preprocessed_clean['year_of_publication'] > 2025), 
    'year_of_publication'
] = np.nan

# Imputasi dengan median
median_year = preprocessed_clean['year_of_publication'].median()
preprocessed_clean['year_of_publication'] = preprocessed_clean['year_of_publication'].fillna(median_year)


# Kode ini membersihkan kolom `year_of_publication` dengan:
# 1. Mengubah tipe data ke `float`,
# 2. Mengganti nilai tahun yang tidak masuk akal (<1500 atau >2025) menjadi NaN,
# 3. Mengisi nilai NaN tersebut dengan nilai median dari tahun publikasi yang valid.

# In[24]:


from ast import literal_eval

def safe_literal_eval(val):
    if isinstance(val, str):
        try:
            result = literal_eval(val)
            return result if isinstance(result, list) else []
        except:
            return []
    return []

preprocessed_clean['Category'] = preprocessed_clean['Category'].apply(safe_literal_eval)


# Kode ini mengubah nilai kolom `Category` dari string seperti `['Fiction']` menjadi list Python `['Fiction']`. Jika gagal dievaluasi atau formatnya tidak sesuai, akan dikembalikan sebagai list kosong [].
# 

# In[26]:


# Cek jumlah missing per kolom
print(preprocessed_clean.isnull().sum())


# In[27]:


# Isi city kosong dengan string kosong
preprocessed_clean['city'] = preprocessed_clean['city'].fillna('')

# Isi state kosong dengan kategori "Unknown"
preprocessed_clean['state'] = preprocessed_clean['state'].fillna("['Unknown']")

# Isi country kosong dengan 'unknown'
preprocessed_clean['country'] = preprocessed_clean['country'].fillna('unknown')

# Isi book_author kosong dengan 'unknown'
preprocessed_clean['book_author'] = preprocessed_clean['book_author'].fillna('unknown')


# In[28]:


# Cek jumlah baris dan kolom
print(f"Jumlah baris: {preprocessed_clean.shape[0]}")
print(f"Jumlah kolom: {preprocessed_clean.shape[1]}")

# Lihat struktur data
preprocessed_clean.info()

# Statistik deskriptif numerik
preprocessed_clean.describe()


# Dataset telah dibersihkan dari nilai kosong pada kolom city, state, country, dan book_author dengan mengganti nilai kosong menggunakan string default seperti '', ['Unknown'], dan 'unknown'. Setelah itu, jumlah total data adalah 1.028.027 baris dan 18 kolom. Struktur data terdiri dari kombinasi tipe int, float, dan object. Berdasarkan statistik deskriptif:
# 
# * Rating buku adalah 0-10.
# * Tahun terbit buku berkisar dari 1806 hingga 2008.
# Ini menunjukkan data siap digunakan untuk analisis lanjutan atau pemodelan sistem rekomendasi.

# In[30]:


# Jumlah missing values per kolom
missing = preprocessed_clean.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("Missing Values:\n", missing)


# Dari hasil diatas, tidak ada lagi missing values di dataset, artinya semua data sudah lengkap dan siap digunakan untuk analisis atau pemodelan.

# # **Exploratory Data Analysis (EDA)**

# ### 1. Distribusi Nilai Rating

# In[34]:


plt.figure(figsize=(8, 5))
sns.countplot(data=preprocessed_clean, x='rating', hue='rating', palette='viridis', legend=False)
plt.title("Distribusi Nilai Rating")
plt.xlabel("Rating")
plt.ylabel("Jumlah")
plt.show()


# Berdasarkan grafik distribusi nilai rating:
# * Rating 0 mendominasi jumlah data secara signifikan, dengan lebih dari 600.000 entri. Ini kemungkinan besar menunjukkan rating implisit atau tidak ada rating yang diberikan oleh pengguna.
# * Rating dari 1 hingga 10 jumlahnya jauh lebih sedikit, dengan puncak pada rating 8, diikuti oleh rating 7, 10, dan 5.
# * Distribusi rating menunjukkan bahwa ketika pengguna memberikan rating eksplisit, mereka cenderung memberikan nilai tinggi (positif).
# 
# Kesimpulan:
# Mayoritas data berisi rating 0 (bisa dianggap sebagai "tidak diketahui" atau "tidak diberikan"), dan data rating eksplisit menunjukkan kecenderungan bias positif. Hal ini perlu dipertimbangkan dalam pemodelan, misalnya dengan memisahkan rating 0 dari rating eksplisit.

# ### 2. Top 10 Buku dengan Rating Terbanyak

# In[37]:


top_books = preprocessed_clean['book_title'].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_books.values, y=top_books.index, hue=top_books.index, palette='magma', legend=False)
plt.title("Top 10 Buku dengan Rating Terbanyak")
plt.xlabel("Jumlah Rating")
plt.ylabel("Judul Buku")
plt.show()


# Berdasarkan grafik Top 10 Buku dengan Rating Terbanyak:
# * Buku "Wild Animus" memperoleh jumlah rating terbanyak secara signifikan dibandingkan buku lain, melebihi 2.500 rating.
# * Buku populer lain seperti "The Lovely Bones: A Novel", "The Da Vinci Code", dan "The Secret Life of Bees" juga masuk dalam daftar dengan jumlah rating yang tinggi.
# * Grafik ini mencerminkan buku-buku yang paling sering dinilai oleh pengguna, bukan buku dengan rating tertinggi secara kualitas.
# 
# Kesimpulan:
# Buku-buku pada grafik ini bisa dianggap sebagai buku populer atau paling dikenal oleh komunitas pengguna, sehingga cocok dijadikan acuan awal untuk sistem rekomendasi berbasis popularitas atau cold-start (tanpa data pengguna).

# ### 3. Distribusi Usia Pengguna

# In[40]:


plt.figure(figsize=(8, 5))
sns.histplot(preprocessed_clean['age'], bins=30, kde=True, color='skyblue')
plt.title("Distribusi Usia Pengguna")
plt.xlabel("Usia")
plt.ylabel("Jumlah")
plt.show()


# Berdasarkan grafik Distribusi Usia Pengguna:
# * Mayoritas pengguna berada pada rentang usia 30–40 tahun, dengan puncak tertinggi sekitar usia 35 tahun.
# * Terlihat adanya penurunan tajam di luar rentang usia tersebut, terutama setelah usia 60 tahun.
# * Distribusi menunjukkan pola right-skewed, artinya lebih banyak pengguna muda hingga paruh baya dibandingkan pengguna lansia.
# 
# Kesimpulan:
# Sistem rekomendasi buku sebaiknya mempertimbangkan dominasi usia 30–40 tahun ini, karena preferensi mereka kemungkinan besar paling merepresentasikan tren umum dalam data.

# ### 4. Top 10 Bahasa Buku Terpopuler

# In[43]:


top_lang = preprocessed_clean['Language'].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_lang.values, y=top_lang.index, hue=top_lang.index, palette='coolwarm', legend=False)
plt.title("Top 10 Bahasa Buku Terpopuler")
plt.xlabel("Jumlah Buku")
plt.ylabel("Bahasa")
plt.show()


# Grafik Top 10 Bahasa Buku Terpopuler menunjukkan distribusi bahasa dari buku-buku yang tersedia dalam dataset:
# 
# * Bahasa Inggris (en) mendominasi secara signifikan dengan jumlah buku terbanyak (lebih dari 600.000 judul).
# * Label ‘9’ muncul sebagai bahasa kedua terbanyak, yang kemungkinan besar merupakan data anomali atau kode yang salah dalam pengisian kolom bahasa.
# * Bahasa lainnya seperti Jerman (de), Spanyol (es), Prancis (fr), dan beberapa bahasa Eropa lainnya muncul dalam jumlah yang jauh lebih kecil.
# 
# Kesimpulan:
# Mayoritas buku dalam dataset berbahasa Inggris, sehingga sistem rekomendasi kemungkinan akan lebih relevan bagi pengguna yang memahami bahasa tersebut. Perlu penanganan khusus terhadap entri bahasa tidak valid seperti ‘9’.

# ### 5. Top 10 Kategori Buku

# In[46]:


all_categories = [cat.strip().lower() for sublist in preprocessed_clean['Category'] for cat in sublist]
cat_series = pd.Series(all_categories)
top_categories = cat_series.value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(
    x=top_categories.values,
    y=top_categories.index,
    hue=top_categories.index,
    palette='Set2',
    legend=False
)
plt.title("Top 10 Kategori Buku")
plt.xlabel("Jumlah")
plt.ylabel("Kategori")
plt.show()


# Grafik Top 10 Kategori Buku menunjukkan distribusi kategori buku yang paling banyak terdapat dalam dataset:
# 
# * Kategori "Fiction" mendominasi secara signifikan, dengan jumlah buku hampir 400.000 judul. Ini menunjukkan bahwa fiksi adalah genre yang paling umum dan populer dalam koleksi data ini.
# *Diikuti oleh "Juvenile Fiction" (fiksi untuk remaja) dan "Biography & Autobiography", yang juga memiliki jumlah signifikan namun jauh lebih kecil dibandingkan fiksi umum.
# * Kategori lain seperti "Humor", "History", dan "Religion" muncul dalam jumlah lebih terbatas.
# 
# Kesimpulan: Genre fiksi mendominasi isi dataset, sehingga sistem rekomendasi kemungkinan akan lebih banyak merekomendasikan buku-buku dalam kategori ini kecuali dilakukan penyesuaian khusus untuk genre lainnya.

# # **Modelling**

# # **Content-Based Filtering**

# ### A. Data Preparation

# Dalam pendekatan Content-Based Filtering, sistem rekomendasi akan memberikan saran buku yang mirip secara konten dengan buku yang pernah disukai pengguna. Fokus utama metode ini adalah menganalisis informasi deskriptif dari setiap item (dalam hal ini, buku) untuk menemukan kemiripan antar buku berdasarkan atribut-atribut berikut:
# 
# 📖 Summary: Ringkasan isi buku yang menggambarkan topik atau cerita.
# 
# 🏷️ Category: Kategori atau genre buku, misalnya [Fiction], [Science], [Biography], dll.
# 
# 📘 book_title: Judul buku, karena bisa mengandung kata-kata penting terkait isi.
# 
# 🏢 publisher: Nama penerbit, yang dalam beberapa kasus mengindikasikan genre atau kualitas buku.
# 
# 

# In[57]:


preprocessed_clean.head


# In[89]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ✅ 0. Filter data yang tidak valid SEBELUM gabungkan teks

# Hapus entri dengan ringkasan tidak valid (misal: hanya berisi '9')
preprocessed_clean = preprocessed_clean[preprocessed_clean['Summary'].str.strip() != '9']

# Hapus entri dengan kategori kosong (misal: [])
preprocessed_clean = preprocessed_clean[preprocessed_clean['Category'].astype(str) != '[]']

# Hapus entri dengan ringkasan terlalu pendek (< 30 karakter)
preprocessed_clean = preprocessed_clean[preprocessed_clean['Summary'].str.len() > 30]

# Hapus duplikasi berdasarkan book_title + publisher + Summary
preprocessed_clean = preprocessed_clean.drop_duplicates(subset=['book_title', 'publisher', 'Summary'])

# ✅ 1. Gabungkan beberapa kolom jadi satu fitur teks (Summary + Category + book_title + Publisher)
preprocessed_clean['combined'] = (
    preprocessed_clean['Summary'].fillna('') + ' ' +
    preprocessed_clean['Category'].astype(str) + ' ' +
    preprocessed_clean['book_title'].fillna('') + ' ' +
    preprocessed_clean['publisher'].fillna('')
)

# 2. Tampilkan data awal (opsional)
display(preprocessed_clean.head())

# ✅ 3. Hapus duplikasi berdasarkan kolom gabungan (jika masih ada)
preprocessed_clean = preprocessed_clean.drop_duplicates(subset='combined')

# ✅ 3.1 Reset index agar tfidf_matrix sesuai dengan DataFrame
preprocessed_clean = preprocessed_clean.reset_index(drop=True)

# 4. Cek jumlah data setelah menghapus duplikasi
print(f"Jumlah data setelah menghapus duplikasi: {len(preprocessed_clean)}")

# 5. Inisialisasi TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# 6. Transformasi ke matriks TF-IDF
tfidf_matrix = tfidf.fit_transform(preprocessed_clean['combined'])

# 7. Buat model Nearest Neighbors berbasis cosine similarity
nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
nn_model.fit(tfidf_matrix)

# 8. Buat mapping judul → index
indices = pd.Series(preprocessed_clean.index, index=preprocessed_clean['book_title'].str.lower()).drop_duplicates()


# # ✅ Kesimpulan Tahapan Content-Based Filtering
# 
#  1. Pembersihan dan Filter Data
#  - Menghapus entri dengan ringkasan tidak valid ('9')
#  - Menghapus entri dengan kategori kosong ([])
#  - Menghapus ringkasan yang terlalu pendek (< 30 karakter)
#  - Menghapus duplikasi berdasarkan book_title + publisher + Summary
# 
#  2. Penggabungan Fitur Teks
#  - Menggabungkan kolom Summary, Category, book_title, dan publisher ke kolom 'combined'
#  - Ini bertujuan menyatukan informasi penting dalam satu representasi teks yang kaya konten
# 
#  3. Penghapusan Duplikasi dan Reset Index
#  - Membersihkan data dari duplikasi berdasarkan kolom 'combined'
#  - Reset index untuk memastikan baris DataFrame sejajar dengan tfidf_matrix
# 
#  4. TF-IDF Vectorization
#  - Mengubah data teks pada kolom 'combined' menjadi vektor numerik dengan TfidfVectorizer
#  - TF-IDF membantu menekankan kata-kata penting dan menurunkan bobot kata umum
# 
#  5. Pelatihan Model Nearest Neighbors
#  - Melatih model NearestNeighbors dengan tfidf_matrix
#  - Menggunakan cosine similarity untuk mengukur kemiripan antar buku
# 
#  6. Pemetaan Judul Buku ke Index
#  - Membuat mapping dari judul buku (dalam lowercase) ke index DataFrame
#  - Memungkinkan akses langsung ke representasi vektor berdasarkan input judul buku
# 
#  📌 Hasil Akhir
#  Sistem rekomendasi content-based siap digunakan untuk menyarankan buku yang mirip
#  berdasarkan isi dan metadata, tanpa bergantung pada rating pengguna lain.
#  Sangat cocok untuk cold-start scenario seperti buku baru yang belum memiliki rating.
# 

# ## B. Model and Result

# Menemukan buku-buku yang paling mirip dengan buku input, sehingga bisa memberikan rekomendasi yang relevan secara konten, meskipun belum pernah diberi rating oleh pengguna.

# In[93]:


def recommend_books_nn(title, df=preprocessed_clean, top_n=5):
    title = title.lower()

    if title not in indices:
        return f"Buku dengan judul '{title}' tidak ditemukan."

    idx = indices[title]

    # Vektorisasi buku input
    book_vector = tfidf_matrix[idx]

    # Temukan top-N buku mirip
    distances, indices_nn = nn_model.kneighbors(book_vector, n_neighbors=top_n+1)

    # Ambil hasil rekomendasi (kecuali dirinya sendiri)
    rec_indices = indices_nn[0][1:]

    return df[['book_title', 'Category', 'Summary', 'publisher']].iloc[rec_indices]


# 
# Penjelasan Fungsi:
# 1. Fungsi menerima judul buku yang ingin dicari kemiripannya.
# 2. Menggunakan TF-IDF untuk mengukur kemiripan konten antar buku.
# 3. Model NearestNeighbors digunakan untuk mencari buku dengan vektor TF-IDF paling dekat (mirip).
# 4. Mengembalikan beberapa buku yang memiliki konten mirip berdasarkan ringkasan, kategori, judul, dan penerbit.
# 

# ## C. Testing System Recommendation

# In[96]:


recommend_books_nn("The Secret Life of Bees", top_n=5)


# ### Penjelasan Output Rekomendasi: "The Secret Life of Bees"
# 
# Hasil dari pemanggilan fungsi `recommend_books_nn("The Secret Life of Bees", top_n=5)` memberikan 5 buku yang direkomendasikan berdasarkan kemiripan konten dengan buku input. Rekomendasi ini dihasilkan dari model content-based filtering yang membandingkan fitur teks gabungan (summary, kategori, judul, dan penerbit).
# 
# Berikut adalah penjelasan untuk masing-masing buku hasil rekomendasi:
# 
# 1. **Black Boy**
#    - **Kategori**: [African American authors]
#    - **Alasan Rekomendasi**: Buku ini memiliki konten naratif yang kuat tentang pengalaman hidup, mirip dengan tema pencarian identitas dalam *The Secret Life of Bees*.
# 
# 2. **Midnight Heat**
#    - **Kategori**: [Fiction]
#    - **Alasan Rekomendasi**: Novel fiksi dengan elemen emosi kuat dan konflik sosial, cocok dengan pembaca yang menyukai dinamika karakter dan perjuangan batin.
# 
# 3. **Clover**
#    - **Kategori**: [Fiction]
#    - **Alasan Rekomendasi**: Sama-sama menceritakan hubungan keluarga dan trauma kehilangan, dengan karakter wanita muda sebagai tokoh utama.
# 
# 4. **LIBRARY OF CLASSIC CHILDREN'S LITERATURE**
#    - **Kategori**: [Juvenile Fiction]
#    - **Alasan Rekomendasi**: Meskipun ditujukan untuk anak-anak, rekomendasi muncul karena kemiripan gaya naratif dan nilai-nilai emosional yang terkandung dalam cerita.
# 
# 5. **The Secret Life of Bees**
#    - **Kategori**: [Fiction]
#    - **Catatan**: Buku ini sendiri tetap muncul sebagai hasil teratas, karena secara teknis memiliki kemiripan tertinggi dengan dirinya sendiri. Namun hasil akhir hanya menampilkan buku-buku selain buku input, sesuai logika dalam fungsi.
# 
# ### Kesimpulan
# Rekomendasi yang dihasilkan menunjukkan bahwa sistem dapat mengenali pola naratif, tema emosional, serta atribut tekstual lainnya yang relevan. Hal ini membuktikan bahwa pendekatan content-based cukup efektif meskipun tidak menggunakan data rating pengguna.
# 

# In[99]:


recommend_books_nn("A Painted House", top_n=5)


# ### 📚 Recommendation Output
# 
# #### Judul: *A Painted House*
# 
# Berdasarkan pendekatan content-based filtering, sistem memberikan rekomendasi buku yang memiliki kemiripan konten dengan *A Painted House* melalui analisis terhadap *summary*, *category*, *judul*, dan *publisher*. Berikut penjelasan hasil rekomendasinya:
# 
# 1. **A Painted House (Delta)** — Buku utama dan referensi pencarian. Kisahnya mengangkat tema ketegangan rasial dan cinta terlarang, dengan latar kehidupan petani.
# 
# 2. **A Painted House (Limited Edition) - Doubleday** — Edisi berbeda dari buku yang sama, tetap relevan karena memiliki isi konten dan cerita serupa.
# 
# 3. **Boy of the Painted Cave** — Memiliki elemen tematik serupa seperti pembatasan sosial dan perjuangan hidup di masyarakat tertutup, ditulis dalam genre *Juvenile Fiction*.
# 
# 4. **River of Earth** — Cerita tentang keluarga miskin yang berjuang untuk kehidupan lebih baik, serupa dengan latar ekonomi dan sosial di *A Painted House*.
# 
# 5. **The Wedding Dress** — Menceritakan potret kehidupan perintis dengan gaya naratif yang mirip, menggambarkan masa lalu dan perjuangan hidup di era tertentu.
# 
# #### Kesimpulan:
# Rekomendasi ini menunjukkan bahwa sistem berhasil mengidentifikasi buku-buku dengan kemiripan tema, genre, dan latar suasana, bahkan meskipun judul atau pengarang berbeda. Ini menegaskan keefektifan pendekatan content-based dalam menemukan buku sejenis berdasarkan isi konten, bukan hanya popularitas atau rating pengguna lain.
# 

# In[ ]:




