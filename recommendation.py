{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "3d06b822",
   "metadata": {},
   "source": [
    "# **SYSTEM RECOMMENDATION : BOOK RECOMMENDATION SYSTEM**\n",
    "## Nama : Deawi Guna Pratiwi\n",
    "## Email : deawigunapratiwi@gmail.com\n",
    "## Sumber Dataset :\n",
    "Dataset diperoleh dari kaggle dengan judul **Book-Crossing: User review ratings** (https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset) dengan jumlah dataset 1031175 data."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c92cd466",
   "metadata": {},
   "source": [
    "# **Proyek Overview**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "09f03287",
   "metadata": {},
   "source": [
    "Di era digital, informasi mengenai buku sangat melimpah. Meskipun hal ini memberikan banyak pilihan bagi pembaca, pada saat yang sama muncul tantangan besar dalam menemukan buku yang benar-benar relevan dengan minat dan preferensi masing-masing individu. Pengguna kerap mengalami kebingungan dalam menentukan pilihan, terutama ketika dihadapkan dengan ribuan judul buku yang tersedia secara online. Untuk mengatasi permasalahan ini, dibutuhkan sebuah sistem rekomendasi yang cerdas dan efisien, yang mampu menyarankan buku secara personal berdasarkan data dan perilaku pengguna sebelumnya.\n",
    "\n",
    "Book-Crossing Dataset, yang diperoleh dari komunitas Book-Crossing, menjadi salah satu sumber data yang ideal dalam membangun sistem seperti itu. Dataset ini mencakup lebih dari satu juta data interaksi pengguna berupa rating terhadap buku, serta dilengkapi metadata buku dan informasi dasar pengguna. Dengan struktur data yang lengkap, dataset ini sangat cocok untuk diterapkan pada berbagai pendekatan sistem rekomendasi, seperti content-based filtering, collaborative filtering, maupun pendekatan hybrid yang menggabungkan keduanya.\n",
    "\n",
    "Tujuan dari proyek ini adalah untuk membangun sistem rekomendasi buku yang mampu memberikan saran yang akurat dan personal kepada pengguna. Selain itu, proyek ini juga bertujuan untuk menerapkan dan membandingkan beberapa pendekatan sistem rekomendasi guna mengevaluasi efektivitas masing-masing metode. Melalui pemanfaatan data rating dan metadata buku, diharapkan sistem ini mampu mengidentifikasi pola preferensi pengguna serta membantu mereka dalam menemukan buku yang sesuai dengan minat mereka dengan lebih mudah dan efisien."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ef96d65e",
   "metadata": {},
   "source": [
    "# 💼 **Business Understanding**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "984bf7da",
   "metadata": {},
   "source": [
    "## 🔍 **Problem Statements**\n",
    "1. Banyak pengguna mengalami kesulitan dalam menemukan buku yang sesuai preferensinya karena belum tersedia sistem rekomendasi yang personal.\n",
    "2. Platform belum memanfaatkan data interaksi pengguna dan metadata buku secara optimal untuk menghasilkan rekomendasi.\n",
    "3. Belum dilakukan analisis mendalam terhadap data pengguna dan buku untuk memahami pola dan tren yang mendukung pengambilan keputusan sistem rekomendasi.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "daf53d1d",
   "metadata": {},
   "source": [
    "## 🎯 **Goals**\n",
    "1. Mengembangkan sistem rekomendasi buku yang mampu menyarankan judul secara personal berdasarkan data pengguna dan metadata buku.\n",
    "2. Melakukan exploratory data analysis (EDA) untuk memahami struktur data, distribusi, dan hubungan antar variabel.\n",
    "3. Menerapkan algoritma content-based filtering untuk memberikan rekomendasi buku berdasarkan kemiripan fitur konten buku."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ffe87351",
   "metadata": {},
   "source": [
    "## 🛠️ **Solution Approach**\n",
    "1. Melakukan Exploratory Data Analysis (EDA) untuk memahami distribusi data, mendeteksi anomali, serta mengeksplorasi pola interaksi pengguna dan karakteristik buku melalui visualisasi.\n",
    "\n",
    "2. Membangun sistem content-based filtering dengan memanfaatkan metadata buku (judul, penulis, penerbit) dan menghitung tingkat kesamaan antar buku menggunakan algoritma cosine similarity.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a3451032",
   "metadata": {},
   "source": [
    "# **Data Understanding**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a379a4a5",
   "metadata": {},
   "source": [
    "## **Import Library**\n",
    "Import semua library yang diperlukan"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "8ed53e2a",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "import os\n",
    "import shutil\n",
    "import zipfile\n",
    "import re\n",
    "from IPython.display import display\n",
    "import textwrap\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.ticker as mtick\n",
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "from tensorflow.keras import layers\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "b610b2fa",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "📁 Folder sudah ada, ekstraksi dilewati.\n"
     ]
    }
   ],
   "source": [
    "#membuka zip menjadi folder\n",
    "# Path file zip yang kamu upload\n",
    "zip_path = \"archive.zip\"\n",
    "extract_to = \"/bookcrossing_data\"\n",
    "\n",
    "# Mengekstrak jika folder belum ada\n",
    "if not os.path.exists(extract_to):\n",
    "    with zipfile.ZipFile(zip_path, 'r') as zip_ref:\n",
    "        zip_ref.extractall(extract_to)\n",
    "    print(f\"✅ Berhasil mengekstrak ke: {extract_to}\")\n",
    "else:\n",
    "    print(\"📁 Folder sudah ada, ekstraksi dilewati.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "6b033b76",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "📁 Folder: /bookcrossing_data\n",
      "\n",
      "📁 Folder: /bookcrossing_data\\Book reviews\n",
      "\n",
      "📁 Folder: /bookcrossing_data\\Book reviews\\Book reviews\n",
      "  - BX-Book-Ratings.csv\n",
      "  - BX-Users.csv\n",
      "  - BX_Books.csv\n",
      "\n",
      "📁 Folder: /bookcrossing_data\\Books Data with Category Language and Summary\n",
      "  - Preprocessed_data.csv\n"
     ]
    }
   ],
   "source": [
    "# Menelusuri semua file dan subfolder\n",
    "for root, dirs, files in os.walk(extract_to):\n",
    "    print(f\"\\n📁 Folder: {root}\")\n",
    "    for f in files:\n",
    "        print(\"  -\", f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "ece8e974-819a-44db-bc1a-f771a9400dee",
   "metadata": {},
   "outputs": [],
   "source": [
    "# load file dari folder Books Data with Category Language and Summary\n",
    "preprocessed = pd.read_csv(f\"{extract_to}/Books Data with Category Language and Summary/Preprocessed_data.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "24bc2e49-3870-412a-b5cc-9a64110f294b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>user_id</th>\n",
       "      <th>location</th>\n",
       "      <th>age</th>\n",
       "      <th>isbn</th>\n",
       "      <th>rating</th>\n",
       "      <th>book_title</th>\n",
       "      <th>book_author</th>\n",
       "      <th>year_of_publication</th>\n",
       "      <th>publisher</th>\n",
       "      <th>img_s</th>\n",
       "      <th>img_m</th>\n",
       "      <th>img_l</th>\n",
       "      <th>Summary</th>\n",
       "      <th>Language</th>\n",
       "      <th>Category</th>\n",
       "      <th>city</th>\n",
       "      <th>state</th>\n",
       "      <th>country</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>stockton, california, usa</td>\n",
       "      <td>18.0000</td>\n",
       "      <td>0195153448</td>\n",
       "      <td>0</td>\n",
       "      <td>Classical Mythology</td>\n",
       "      <td>Mark P. O. Morford</td>\n",
       "      <td>2002.0</td>\n",
       "      <td>Oxford University Press</td>\n",
       "      <td>http://images.amazon.com/images/P/0195153448.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0195153448.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0195153448.0...</td>\n",
       "      <td>Provides an introduction to classical myths pl...</td>\n",
       "      <td>en</td>\n",
       "      <td>['Social Science']</td>\n",
       "      <td>stockton</td>\n",
       "      <td>california</td>\n",
       "      <td>usa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>8</td>\n",
       "      <td>timmins, ontario, canada</td>\n",
       "      <td>34.7439</td>\n",
       "      <td>0002005018</td>\n",
       "      <td>5</td>\n",
       "      <td>Clara Callan</td>\n",
       "      <td>Richard Bruce Wright</td>\n",
       "      <td>2001.0</td>\n",
       "      <td>HarperFlamingo Canada</td>\n",
       "      <td>http://images.amazon.com/images/P/0002005018.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0002005018.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0002005018.0...</td>\n",
       "      <td>In a small town in Canada, Clara Callan reluct...</td>\n",
       "      <td>en</td>\n",
       "      <td>['Actresses']</td>\n",
       "      <td>timmins</td>\n",
       "      <td>ontario</td>\n",
       "      <td>canada</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>11400</td>\n",
       "      <td>ottawa, ontario, canada</td>\n",
       "      <td>49.0000</td>\n",
       "      <td>0002005018</td>\n",
       "      <td>0</td>\n",
       "      <td>Clara Callan</td>\n",
       "      <td>Richard Bruce Wright</td>\n",
       "      <td>2001.0</td>\n",
       "      <td>HarperFlamingo Canada</td>\n",
       "      <td>http://images.amazon.com/images/P/0002005018.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0002005018.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0002005018.0...</td>\n",
       "      <td>In a small town in Canada, Clara Callan reluct...</td>\n",
       "      <td>en</td>\n",
       "      <td>['Actresses']</td>\n",
       "      <td>ottawa</td>\n",
       "      <td>ontario</td>\n",
       "      <td>canada</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>11676</td>\n",
       "      <td>n/a, n/a, n/a</td>\n",
       "      <td>34.7439</td>\n",
       "      <td>0002005018</td>\n",
       "      <td>8</td>\n",
       "      <td>Clara Callan</td>\n",
       "      <td>Richard Bruce Wright</td>\n",
       "      <td>2001.0</td>\n",
       "      <td>HarperFlamingo Canada</td>\n",
       "      <td>http://images.amazon.com/images/P/0002005018.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0002005018.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0002005018.0...</td>\n",
       "      <td>In a small town in Canada, Clara Callan reluct...</td>\n",
       "      <td>en</td>\n",
       "      <td>['Actresses']</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>41385</td>\n",
       "      <td>sudbury, ontario, canada</td>\n",
       "      <td>34.7439</td>\n",
       "      <td>0002005018</td>\n",
       "      <td>0</td>\n",
       "      <td>Clara Callan</td>\n",
       "      <td>Richard Bruce Wright</td>\n",
       "      <td>2001.0</td>\n",
       "      <td>HarperFlamingo Canada</td>\n",
       "      <td>http://images.amazon.com/images/P/0002005018.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0002005018.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0002005018.0...</td>\n",
       "      <td>In a small town in Canada, Clara Callan reluct...</td>\n",
       "      <td>en</td>\n",
       "      <td>['Actresses']</td>\n",
       "      <td>sudbury</td>\n",
       "      <td>ontario</td>\n",
       "      <td>canada</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0  user_id                   location      age        isbn  \\\n",
       "0           0        2  stockton, california, usa  18.0000  0195153448   \n",
       "1           1        8   timmins, ontario, canada  34.7439  0002005018   \n",
       "2           2    11400    ottawa, ontario, canada  49.0000  0002005018   \n",
       "3           3    11676              n/a, n/a, n/a  34.7439  0002005018   \n",
       "4           4    41385   sudbury, ontario, canada  34.7439  0002005018   \n",
       "\n",
       "   rating           book_title           book_author  year_of_publication  \\\n",
       "0       0  Classical Mythology    Mark P. O. Morford               2002.0   \n",
       "1       5         Clara Callan  Richard Bruce Wright               2001.0   \n",
       "2       0         Clara Callan  Richard Bruce Wright               2001.0   \n",
       "3       8         Clara Callan  Richard Bruce Wright               2001.0   \n",
       "4       0         Clara Callan  Richard Bruce Wright               2001.0   \n",
       "\n",
       "                 publisher                                              img_s  \\\n",
       "0  Oxford University Press  http://images.amazon.com/images/P/0195153448.0...   \n",
       "1    HarperFlamingo Canada  http://images.amazon.com/images/P/0002005018.0...   \n",
       "2    HarperFlamingo Canada  http://images.amazon.com/images/P/0002005018.0...   \n",
       "3    HarperFlamingo Canada  http://images.amazon.com/images/P/0002005018.0...   \n",
       "4    HarperFlamingo Canada  http://images.amazon.com/images/P/0002005018.0...   \n",
       "\n",
       "                                               img_m  \\\n",
       "0  http://images.amazon.com/images/P/0195153448.0...   \n",
       "1  http://images.amazon.com/images/P/0002005018.0...   \n",
       "2  http://images.amazon.com/images/P/0002005018.0...   \n",
       "3  http://images.amazon.com/images/P/0002005018.0...   \n",
       "4  http://images.amazon.com/images/P/0002005018.0...   \n",
       "\n",
       "                                               img_l  \\\n",
       "0  http://images.amazon.com/images/P/0195153448.0...   \n",
       "1  http://images.amazon.com/images/P/0002005018.0...   \n",
       "2  http://images.amazon.com/images/P/0002005018.0...   \n",
       "3  http://images.amazon.com/images/P/0002005018.0...   \n",
       "4  http://images.amazon.com/images/P/0002005018.0...   \n",
       "\n",
       "                                             Summary Language  \\\n",
       "0  Provides an introduction to classical myths pl...       en   \n",
       "1  In a small town in Canada, Clara Callan reluct...       en   \n",
       "2  In a small town in Canada, Clara Callan reluct...       en   \n",
       "3  In a small town in Canada, Clara Callan reluct...       en   \n",
       "4  In a small town in Canada, Clara Callan reluct...       en   \n",
       "\n",
       "             Category      city       state country  \n",
       "0  ['Social Science']  stockton  california     usa  \n",
       "1       ['Actresses']   timmins     ontario  canada  \n",
       "2       ['Actresses']    ottawa     ontario  canada  \n",
       "3       ['Actresses']       NaN         NaN     NaN  \n",
       "4       ['Actresses']   sudbury     ontario  canada  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Display the first few rows\n",
    "preprocessed.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4332215e-662b-4eb4-a496-97e41a6eec93",
   "metadata": {},
   "source": [
    "## **Deskripsi Variabel**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dc55fa0d-ab10-4214-90ab-25ad8b87ff04",
   "metadata": {},
   "source": [
    "| **Nama Variabel**     | **Tipe Data** | **Deskripsi**                                                               |\n",
    "| --------------------- | ------------- | --------------------------------------------------------------------------- |\n",
    "| `Unnamed: 0`          | integer       | Indeks baris otomatis dari proses sebelumnya (dapat diabaikan atau dihapus) |\n",
    "| `user_id`             | integer       | ID unik dari pengguna yang memberikan rating                                |\n",
    "| `location`            | string        | Lokasi pengguna dalam format `kota, provinsi, negara`                       |\n",
    "| `age`                 | float         | Usia pengguna                                                               |\n",
    "| `isbn`                | string        | Nomor identifikasi unik buku (International Standard Book Number)           |\n",
    "| `rating`              | float/integer | Nilai rating yang diberikan pengguna terhadap buku (biasanya dari 0–10)     |\n",
    "| `book_title`          | string        | Judul buku                                                                  |\n",
    "| `book_author`         | string        | Nama penulis buku                                                           |\n",
    "| `year_of_publication` | float/integer | Tahun penerbitan buku                                                       |\n",
    "| `publisher`           | string        | Nama penerbit buku                                                          |\n",
    "| `img_s`               | string (URL)  | URL gambar sampul buku ukuran kecil                                         |\n",
    "| `img_m`               | string (URL)  | URL gambar sampul buku ukuran sedang                                        |\n",
    "| `img_l`               | string (URL)  | URL gambar sampul buku ukuran besar                                         |\n",
    "| `Summary`             | string        | Ringkasan atau sinopsis dari isi buku                                       |\n",
    "| `Language`            | string        | Bahasa dari isi buku (contoh: `en` untuk English, `fr` untuk French)        |\n",
    "| `Category`            | string/list   | Kategori atau genre buku (dalam format list/array teks)                     |\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "e36fa6e1-0667-4b38-bc67-d92fde963c66",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1031175 entries, 0 to 1031174\n",
      "Data columns (total 19 columns):\n",
      " #   Column               Non-Null Count    Dtype  \n",
      "---  ------               --------------    -----  \n",
      " 0   Unnamed: 0           1031175 non-null  int64  \n",
      " 1   user_id              1031175 non-null  int64  \n",
      " 2   location             1031175 non-null  object \n",
      " 3   age                  1031175 non-null  float64\n",
      " 4   isbn                 1031175 non-null  object \n",
      " 5   rating               1031175 non-null  int64  \n",
      " 6   book_title           1031175 non-null  object \n",
      " 7   book_author          1031174 non-null  object \n",
      " 8   year_of_publication  1031175 non-null  float64\n",
      " 9   publisher            1031175 non-null  object \n",
      " 10  img_s                1031175 non-null  object \n",
      " 11  img_m                1031175 non-null  object \n",
      " 12  img_l                1031175 non-null  object \n",
      " 13  Summary              1031175 non-null  object \n",
      " 14  Language             1031175 non-null  object \n",
      " 15  Category             1031175 non-null  object \n",
      " 16  city                 1017072 non-null  object \n",
      " 17  state                1008377 non-null  object \n",
      " 18  country              995801 non-null   object \n",
      "dtypes: float64(2), int64(3), object(14)\n",
      "memory usage: 149.5+ MB\n"
     ]
    }
   ],
   "source": [
    "preprocessed.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "56b5ba28-50b4-444a-aea7-d012ca6401b1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Total of rows: 1031175\n",
      "Total of column: 19\n"
     ]
    }
   ],
   "source": [
    "# Menampilkan jumlah baris dan kolom pada data\n",
    "total_row, total_column = preprocessed.shape\n",
    "print(f\"Total of rows: {total_row}\")\n",
    "print(f\"Total of column: {total_column}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "35025864-64f2-4c96-ad9c-9a4637d3b2fc",
   "metadata": {},
   "source": [
    "Dapat dilihat bahwa data yang digunakan adalah sebanyak 1031175 data dengan 19 fitur dengan terdapat 2 variabel bertipe `float64`, 3 variabel bertipe `int64`, 14 variabel bertipe `object`."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "828867a7-4809-4c4c-a4eb-e26332941e17",
   "metadata": {},
   "source": [
    "## Statistik Deskripsi dari Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "59b947d4-c195-4247-b3ec-cd24e2914144",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>user_id</th>\n",
       "      <th>age</th>\n",
       "      <th>rating</th>\n",
       "      <th>year_of_publication</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>1.031175e+06</td>\n",
       "      <td>1.031175e+06</td>\n",
       "      <td>1.031175e+06</td>\n",
       "      <td>1.031175e+06</td>\n",
       "      <td>1.031175e+06</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>5.155870e+05</td>\n",
       "      <td>1.405944e+05</td>\n",
       "      <td>3.642902e+01</td>\n",
       "      <td>2.839022e+00</td>\n",
       "      <td>1.995283e+03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>2.976747e+05</td>\n",
       "      <td>8.052444e+04</td>\n",
       "      <td>1.035354e+01</td>\n",
       "      <td>3.854149e+00</td>\n",
       "      <td>7.309340e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>2.000000e+00</td>\n",
       "      <td>5.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>1.376000e+03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>2.577935e+05</td>\n",
       "      <td>7.041500e+04</td>\n",
       "      <td>3.100000e+01</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>1.992000e+03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>5.155870e+05</td>\n",
       "      <td>1.412100e+05</td>\n",
       "      <td>3.474390e+01</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>1.997000e+03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>7.733805e+05</td>\n",
       "      <td>2.114260e+05</td>\n",
       "      <td>4.100000e+01</td>\n",
       "      <td>7.000000e+00</td>\n",
       "      <td>2.001000e+03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1.031174e+06</td>\n",
       "      <td>2.788540e+05</td>\n",
       "      <td>9.900000e+01</td>\n",
       "      <td>1.000000e+01</td>\n",
       "      <td>2.008000e+03</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Unnamed: 0       user_id           age        rating  \\\n",
       "count  1.031175e+06  1.031175e+06  1.031175e+06  1.031175e+06   \n",
       "mean   5.155870e+05  1.405944e+05  3.642902e+01  2.839022e+00   \n",
       "std    2.976747e+05  8.052444e+04  1.035354e+01  3.854149e+00   \n",
       "min    0.000000e+00  2.000000e+00  5.000000e+00  0.000000e+00   \n",
       "25%    2.577935e+05  7.041500e+04  3.100000e+01  0.000000e+00   \n",
       "50%    5.155870e+05  1.412100e+05  3.474390e+01  0.000000e+00   \n",
       "75%    7.733805e+05  2.114260e+05  4.100000e+01  7.000000e+00   \n",
       "max    1.031174e+06  2.788540e+05  9.900000e+01  1.000000e+01   \n",
       "\n",
       "       year_of_publication  \n",
       "count         1.031175e+06  \n",
       "mean          1.995283e+03  \n",
       "std           7.309340e+00  \n",
       "min           1.376000e+03  \n",
       "25%           1.992000e+03  \n",
       "50%           1.997000e+03  \n",
       "75%           2.001000e+03  \n",
       "max           2.008000e+03  "
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "preprocessed.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "431d9c99-99df-4304-a47c-758897fa52bb",
   "metadata": {},
   "source": [
    "Tabel di atas memberikan informasi statistik pada masing-masing kolom, antara lain:\n",
    "- Count adalah jumlah sampel pada data.\n",
    "- Mean adalah nilai rata-rata.\n",
    "- Std adalah standar deviasi (mengukur seberapa tersebar data).\n",
    "- Min yaitu nilai minimum setiap kolom.\n",
    "- 25% adalah kuartil pertama, yaitu nilai di bawah 25% data berada.\n",
    "- 50% adalah kuartil kedua, juga disebut median (nilai tengah data).\n",
    "- 75% adalah kuartil ketiga, yaitu nilai di bawah 75% data berada.\n",
    "- Max adalah nilai maksimum"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4d6bf05e-5533-49f9-bd54-33fe877d490a",
   "metadata": {},
   "source": [
    "Penjelasan:\n",
    "\n",
    "| **Kolom**             | **Penjelasan Statistik**                                                                                                                                                                                                                                                            |\n",
    "| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |\n",
    "| `Unnamed: 0`          | Ini adalah indeks baris otomatis yang di-generate dari proses sebelumnya. Nilainya unik dan tidak penting untuk analisis.                                                                                                                                                           |\n",
    "| `user_id`             | ID pengguna bersifat numerik. Nilai `min = 2` dan `max = 278854`, menunjukkan jumlah pengguna aktif yang cukup besar. Tidak ada indikasi kejanggalan di sini.                                                                                                                       |\n",
    "| `age`                 | Usia pengguna berkisar antara **5 hingga 99 tahun**. Rata-rata usia pengguna adalah sekitar **36 tahun**, dengan standar deviasi sekitar **10 tahun**, artinya mayoritas pengguna berada dalam rentang **25–45 tahun**.                                                             |\n",
    "| `rating`              | Skor rating dari pengguna terhadap buku berada dalam skala **0 hingga 10**. Nilai **mean-nya hanya 2.83**, dan nilai median **0**, menunjukkan bahwa mayoritas pengguna tidak memberikan rating aktif (0 = no rating). Hanya sebagian kecil yang memberi nilai tinggi seperti 7–10. |\n",
    "| `year_of_publication` | Tahun penerbitan buku berkisar dari **1376 hingga 2008**. Nilai minimum (1376) sangat mungkin merupakan outlier (data salah input). Mayoritas buku diterbitkan di sekitar **1992–2001**, dan tahun median-nya adalah **1997**.                                                      |\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "775aa44a-a7b2-449c-9114-f898e832beac",
   "metadata": {},
   "outputs": [],
   "source": [
    "preprocessed_clean = preprocessed.drop(columns=['Unnamed: 0'])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "14b74537-5fce-4349-926a-a651c6b63487",
   "metadata": {},
   "source": [
    "Kode tersebut menghapus kolom `Unnamed: 0` yang tidak dibutuhkan karena biasanya itu hanya duplikat index dari file CSV."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "52fa5834-558a-4ea0-9885-de3d3f19fb4e",
   "metadata": {},
   "outputs": [],
   "source": [
    "preprocessed_clean = preprocessed_clean[preprocessed_clean['age'].between(10, 90)]\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cab9526e-88a1-47d0-be3c-a6c55c2c4beb",
   "metadata": {},
   "source": [
    "kode tersebut menyaring data agar hanya menyertakan pengguna dengan usia antara 10 hingga 90 tahun."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "fc9e3e6d-af50-4d31-9389-833119cdcc99",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Ubah ke integer terlebih dahulu (jika belum)\n",
    "preprocessed_clean['year_of_publication'] = preprocessed_clean['year_of_publication'].astype(float)\n",
    "\n",
    "# Ganti outlier dengan NaN\n",
    "preprocessed_clean.loc[\n",
    "    (preprocessed_clean['year_of_publication'] < 1500) | \n",
    "    (preprocessed_clean['year_of_publication'] > 2025), \n",
    "    'year_of_publication'\n",
    "] = np.nan\n",
    "\n",
    "# Imputasi dengan median\n",
    "median_year = preprocessed_clean['year_of_publication'].median()\n",
    "preprocessed_clean['year_of_publication'] = preprocessed_clean['year_of_publication'].fillna(median_year)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "829bb2d2-a1a7-493c-a543-0e81416047da",
   "metadata": {},
   "source": [
    "Kode ini membersihkan kolom `year_of_publication` dengan:\n",
    "1. Mengubah tipe data ke `float`,\n",
    "2. Mengganti nilai tahun yang tidak masuk akal (<1500 atau >2025) menjadi NaN,\n",
    "3. Mengisi nilai NaN tersebut dengan nilai median dari tahun publikasi yang valid."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "5d925fc4-ab93-49be-8344-d33b27539cc9",
   "metadata": {},
   "outputs": [],
   "source": [
    "from ast import literal_eval\n",
    "\n",
    "def safe_literal_eval(val):\n",
    "    if isinstance(val, str):\n",
    "        try:\n",
    "            result = literal_eval(val)\n",
    "            return result if isinstance(result, list) else []\n",
    "        except:\n",
    "            return []\n",
    "    return []\n",
    "\n",
    "preprocessed_clean['Category'] = preprocessed_clean['Category'].apply(safe_literal_eval)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d1b21c4a-8130-487d-b360-71fa75a2fffc",
   "metadata": {},
   "source": [
    "Kode ini mengubah nilai kolom `Category` dari string seperti `['Fiction']` menjadi list Python `['Fiction']`. Jika gagal dievaluasi atau formatnya tidak sesuai, akan dikembalikan sebagai list kosong [].\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "2d4d0887-9b24-44a6-b40a-791acc5d8481",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "user_id                    0\n",
      "location                   0\n",
      "age                        0\n",
      "isbn                       0\n",
      "rating                     0\n",
      "book_title                 0\n",
      "book_author                1\n",
      "year_of_publication        0\n",
      "publisher                  0\n",
      "img_s                      0\n",
      "img_m                      0\n",
      "img_l                      0\n",
      "Summary                    0\n",
      "Language                   0\n",
      "Category                   0\n",
      "city                   14095\n",
      "state                  22767\n",
      "country                35365\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# Cek jumlah missing per kolom\n",
    "print(preprocessed_clean.isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "7d6d30e6-510a-4eb0-a1db-fc460e9ae56e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Isi city kosong dengan string kosong\n",
    "preprocessed_clean['city'] = preprocessed_clean['city'].fillna('')\n",
    "\n",
    "# Isi state kosong dengan kategori \"Unknown\"\n",
    "preprocessed_clean['state'] = preprocessed_clean['state'].fillna(\"['Unknown']\")\n",
    "\n",
    "# Isi country kosong dengan 'unknown'\n",
    "preprocessed_clean['country'] = preprocessed_clean['country'].fillna('unknown')\n",
    "\n",
    "# Isi book_author kosong dengan 'unknown'\n",
    "preprocessed_clean['book_author'] = preprocessed_clean['book_author'].fillna('unknown')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "57ad928e-c0e1-4663-ad00-78e3d16102a3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Jumlah baris: 1028027\n",
      "Jumlah kolom: 18\n",
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Index: 1028027 entries, 0 to 1031174\n",
      "Data columns (total 18 columns):\n",
      " #   Column               Non-Null Count    Dtype  \n",
      "---  ------               --------------    -----  \n",
      " 0   user_id              1028027 non-null  int64  \n",
      " 1   location             1028027 non-null  object \n",
      " 2   age                  1028027 non-null  float64\n",
      " 3   isbn                 1028027 non-null  object \n",
      " 4   rating               1028027 non-null  int64  \n",
      " 5   book_title           1028027 non-null  object \n",
      " 6   book_author          1028027 non-null  object \n",
      " 7   year_of_publication  1028027 non-null  float64\n",
      " 8   publisher            1028027 non-null  object \n",
      " 9   img_s                1028027 non-null  object \n",
      " 10  img_m                1028027 non-null  object \n",
      " 11  img_l                1028027 non-null  object \n",
      " 12  Summary              1028027 non-null  object \n",
      " 13  Language             1028027 non-null  object \n",
      " 14  Category             1028027 non-null  object \n",
      " 15  city                 1028027 non-null  object \n",
      " 16  state                1028027 non-null  object \n",
      " 17  country              1028027 non-null  object \n",
      "dtypes: float64(2), int64(2), object(14)\n",
      "memory usage: 149.0+ MB\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>user_id</th>\n",
       "      <th>age</th>\n",
       "      <th>rating</th>\n",
       "      <th>year_of_publication</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>1.028027e+06</td>\n",
       "      <td>1.028027e+06</td>\n",
       "      <td>1.028027e+06</td>\n",
       "      <td>1.028027e+06</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>1.403871e+05</td>\n",
       "      <td>3.649525e+01</td>\n",
       "      <td>2.841490e+00</td>\n",
       "      <td>1.995293e+03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>8.042667e+04</td>\n",
       "      <td>1.022372e+01</td>\n",
       "      <td>3.854985e+00</td>\n",
       "      <td>7.253799e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>2.000000e+00</td>\n",
       "      <td>1.000000e+01</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>1.806000e+03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>7.019800e+04</td>\n",
       "      <td>3.100000e+01</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>1.992000e+03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>1.409300e+05</td>\n",
       "      <td>3.474390e+01</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>1.997000e+03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>2.113030e+05</td>\n",
       "      <td>4.100000e+01</td>\n",
       "      <td>7.000000e+00</td>\n",
       "      <td>2.001000e+03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>2.788540e+05</td>\n",
       "      <td>9.000000e+01</td>\n",
       "      <td>1.000000e+01</td>\n",
       "      <td>2.008000e+03</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            user_id           age        rating  year_of_publication\n",
       "count  1.028027e+06  1.028027e+06  1.028027e+06         1.028027e+06\n",
       "mean   1.403871e+05  3.649525e+01  2.841490e+00         1.995293e+03\n",
       "std    8.042667e+04  1.022372e+01  3.854985e+00         7.253799e+00\n",
       "min    2.000000e+00  1.000000e+01  0.000000e+00         1.806000e+03\n",
       "25%    7.019800e+04  3.100000e+01  0.000000e+00         1.992000e+03\n",
       "50%    1.409300e+05  3.474390e+01  0.000000e+00         1.997000e+03\n",
       "75%    2.113030e+05  4.100000e+01  7.000000e+00         2.001000e+03\n",
       "max    2.788540e+05  9.000000e+01  1.000000e+01         2.008000e+03"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Cek jumlah baris dan kolom\n",
    "print(f\"Jumlah baris: {preprocessed_clean.shape[0]}\")\n",
    "print(f\"Jumlah kolom: {preprocessed_clean.shape[1]}\")\n",
    "\n",
    "# Lihat struktur data\n",
    "preprocessed_clean.info()\n",
    "\n",
    "# Statistik deskriptif numerik\n",
    "preprocessed_clean.describe()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a9a12b93-780b-4c54-b861-4ee7a654f0bb",
   "metadata": {},
   "source": [
    "Dataset telah dibersihkan dari nilai kosong pada kolom city, state, country, dan book_author dengan mengganti nilai kosong menggunakan string default seperti '', ['Unknown'], dan 'unknown'. Setelah itu, jumlah total data adalah 1.028.027 baris dan 18 kolom. Struktur data terdiri dari kombinasi tipe int, float, dan object. Berdasarkan statistik deskriptif:\n",
    "\n",
    "* Rating buku adalah 0-10.\n",
    "* Tahun terbit buku berkisar dari 1806 hingga 2008.\n",
    "Ini menunjukkan data siap digunakan untuk analisis lanjutan atau pemodelan sistem rekomendasi."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "bdf708b7-5b41-4b51-ab44-48476629811b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Missing Values:\n",
      " Series([], dtype: int64)\n"
     ]
    }
   ],
   "source": [
    "# Jumlah missing values per kolom\n",
    "missing = preprocessed_clean.isnull().sum()\n",
    "missing = missing[missing > 0].sort_values(ascending=False)\n",
    "print(\"Missing Values:\\n\", missing)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c0566d8e-cdec-4f9a-9f13-d5b2b354ebfa",
   "metadata": {},
   "source": [
    "Dari hasil diatas, tidak ada lagi missing values di dataset, artinya semua data sudah lengkap dan siap digunakan untuk analisis atau pemodelan."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e3b523dc-c686-428c-b6ab-550ad1d0b63f",
   "metadata": {},
   "source": [
    "# **Exploratory Data Analysis (EDA)**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ac2b2e3a-29ee-4425-a12c-fa33791cc66b",
   "metadata": {},
   "source": [
    "### 1. Distribusi Nilai Rating"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "92de5ac6-2324-4d4a-9289-c405b7acf7b7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAtIAAAHUCAYAAAAX288qAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAASUdJREFUeJzt/X9cVHX+///fJ5ARCSZSAacwyYxUNA16KVphq6AtaL7aXW1JEjW20tUIXX/UlmaFPzLaVsuyWnXNYvf1NnZ1TYNsw0xRJChRM3fTRAEpQ1BSUJzPH34930b8xXFwAG/Xy+VcLs45jznncQ7ueufZc55jcTgcDgEAAACol2vc3QAAAADQFBGkAQAAABMI0gAAAIAJBGkAAADABII0AAAAYAJBGgAAADCBIA0AAACYQJAGAAAATCBIAwAAACYQpAE0a0uWLJHFYjG2li1bKigoSPfee69mzZqlsrKyOu+ZMWOGLBZLva7z008/acaMGfr000/r9b5zXatDhw6Ki4ur13lcoUOHDkpMTLykOovFoscee6zOsU8//VQWi0X/7//9P2PfmZ/B3r17jX2JiYnq0KGDqT4v9b2JiYlOP3svLy917NhRkyZNUmVlpalrFxcXa8aMGSooKKhzzMzfGwBNG0EawFVh8eLF2rRpk7KysvTaa6+pR48emjNnjjp37qyPP/7YqfaRRx7Rpk2b6nX+n376Sc8991y9g7SZazWUjIwMPfPMM5dc/84772jXrl0XrYuNjdWmTZvUrl27y2nP8MwzzygjI+OSar29vbVp0yZt2rRJK1eu1L333quXX35Zv/71r01du7i4WM8999w5g3Rj+lkCuDI83d0AAFwJYWFhioiIMF7/6le/0pNPPqm77rpLDzzwgHbv3q3AwEBJ0o033qgbb7yxQfv56aef1KpVqytyrUvVs2fPS66NjIzUjh079NRTT2nFihUXrG3btq3atm17ue0ZOnbseMm111xzjXr37m28HjRokL799ltlZWVpz549CgkJcVlfjelnCeDKYEQawFWrffv2evnll3XkyBG9+eabxv5z/Sf6Tz75RP369VPr1q3l7e2t9u3b61e/+pV++ukn7d271wiKzz33nDGV4Mw0iTPn++KLL/TrX/9a/v7+Rhi80HSAjIwMde/eXS1bttTNN9+sP//5z07HzzVlQvr/T6/4+eh4fn6+4uLiFBAQIKvVKrvdrtjYWO3fv9+oudSpHZJ0/fXXa+rUqfrggw+Uk5Nzwdrz9Xm21157Tffcc48CAgLk4+Ojbt26ae7cuTpx4oRT3eVMC5Fk/EJ18OBBY99//vMfjRo1Sp06dVKrVq10ww03aPDgwdq2bZtR8+mnn+rOO++UJI0aNcr4Oc+YMUPShafprF27VnfccYe8vb1122236S9/+UudvjZs2KDIyEi1bNlSN9xwg5555hm9/fbbl/TsALgHI9IArmq//OUv5eHhofXr15+3Zu/evYqNjdXdd9+tv/zlL7ruuut04MABrV27VjU1NWrXrp3Wrl2rQYMGacyYMXrkkUckqc4o7AMPPKAHH3xQjz32mKqqqi7YV0FBgZKTkzVjxgwFBQVp+fLleuKJJ1RTU6NJkybV6x6rqqoUHR2tkJAQvfbaawoMDFRpaan+/e9/68iRI/U618898cQTWrBggSZPnnzB53ep/vvf/yo+Pl4hISHy8vLSl19+qRdffFFff/31OYOnWXv27JGnp6duvvlmY19xcbFat26t2bNnq23btvrxxx+1dOlS9erVS/n5+QoNDdUdd9yhxYsXa9SoUfrjH/+o2NhYSbroKPSXX36piRMnaurUqQoMDNTbb7+tMWPG6JZbbtE999wjSfrqq68UHR2tW2+9VUuXLlWrVq30xhtv6N1333XZfQNwPYI0gKuaj4+P2rRpo+Li4vPW5OXl6fjx43rppZd0++23G/vj4+ONP4eHh0s6Hap+PpXg50aOHKnnnnvukvoqLi5Wfn6+cb377rtPZWVlev755zV27Fi1atXqks4jSV9//bUOHTqkd955R/fff7+xf9iwYZd8jnPx9vbWjBkzlJSUpH/961+X/QHJtLQ048+nTp3S3XffrdatW2vUqFF6+eWX5e/vb+q8J0+elCRVVFTo//7v//TBBx9o6tSpCggIMGruueceI9RKUm1trWJjY9W1a1e9+eabSktLk5+fn8LCwiSdnl5yvp/z2X744Qd9/vnnat++vXGtdevW6b333jOu+cILL8jDw0Pr1q1TmzZtJJ2eW96tWzdT9wzgymBqB4CrnsPhuODxHj16yMvLS7/73e+0dOlSffvtt6au86tf/eqSa7t27eoU2qXTwb2yslJffPFFva57yy23yN/fX1OmTNEbb7yhHTt21Ov9FzJq1Ch16dJFU6dO1alTpy7rXPn5+RoyZIhat24tDw8PtWjRQg8//LBqa2v1zTffmDpnVVWVWrRooRYtWqhNmzZ6/PHHNXz4cL344otOdSdPnlRqaqq6dOkiLy8veXp6ysvLS7t379bOnTsv67569OhhhGhJatmypW699VZ99913xr7s7Gz94he/MEK0dHp+9+X+sgOgYRGkAVzVqqqqdOjQIdnt9vPWdOzYUR9//LECAgI0btw4dezYUR07dtSrr75ar2vVZ9WKoKCg8+47dOhQva5rs9mUnZ2tHj166KmnnlLXrl1lt9s1ffr0OvOP68vDw0Opqanavn27li5davo8+/bt0913360DBw7o1Vdf1Weffabc3Fy99tprkqRjx46ZOq+3t7dyc3OVm5urVatWqV+/fnr//fc1e/Zsp7qUlBQ988wzGjp0qFatWqXNmzcrNzdXt99+u+lrn9G6des6+6xWq9N5Dx06ZHzY9efOtQ9A48HUDgBXtdWrV6u2tlb9+vW7YN3dd9+tu+++W7W1tdq6davmz5+v5ORkBQYG6sEHH7yka9VnjeHS0tLz7jsTzFq2bClJqq6udqr74Ycf6ry3W7duSk9Pl8Ph0FdffaUlS5Zo5syZ8vb21tSpUy+5r3O5//771bdvX02fPl2LFi0ydY5//OMfqqqq0gcffKCbbrrJ2H+uZebq45prrnFarSU6Olrh4eF67rnn9NBDDyk4OFiS9O677+rhhx9Wamqq0/t/+OEHXXfddZfVw6Vo3bq104cfzzjX3wMAjQcj0gCuWvv27dOkSZNks9n06KOPXtJ7PDw81KtXL2Ok9Mw0C6vVKsn8yOnZtm/fri+//NJp33vvvSdfX1/dcccdkmSsXPHVV1851a1cufK857VYLLr99tv1yiuv6Lrrrqv3NJHzmTNnjoqKiuqsLHKpzvySceY5Sqen3Lz11lsu6e8Mq9Wq1157TcePH9cLL7zgdP2fX1s6/UvWgQMH6rxfct3P+YyoqCh98sknTr8EnTp1Sv/3f//n0usAcC1GpAFcFQoLC3Xy5EmdPHlSZWVl+uyzz7R48WJ5eHgoIyPjguscv/HGG/rkk08UGxur9u3b6/jx48YqEgMGDJAk+fr66qabbtI///lP9e/fX9dff73atGljepk2u92uIUOGaMaMGWrXrp3effddZWVlac6cOcYHDe+8806FhoZq0qRJOnnypPz9/ZWRkaENGzY4netf//qXXn/9dQ0dOlQ333yzHA6HPvjgAx0+fFjR0dGm+jtb3759df/99+uf//ynqfdHR0fLy8tLv/3tbzV58mQdP35cCxcuVHl5uUv6+7moqCj98pe/1OLFizV16lSFhIQoLi5OS5Ys0W233abu3bsrLy9PL730Up0VOTp27Chvb28tX75cnTt31rXXXiu73X7BqUGX4umnn9aqVavUv39/Pf300/L29tYbb7xhrO5yzTWMewGNEf/LBHBVGDVqlCIjI9W/f389/vjjys/P15QpU/T111/r3nvvveB7e/TooZMnT2r69Om67777lJCQoO+//14rV65UTEyMUffOO++oVatWGjJkiO68805jfWEzevToobS0NL388su6//779fnnnystLU2TJ082ajw8PLRq1Srddttteuyxx/Twww/LarVqwYIFTufq1KmTrrvuOs2dO1dDhgzRb37zG33xxRdasmSJkpKSTPd4tlmzZsnDw8PUe2+77TatWLFC5eXleuCBBzR+/Hj16NHD9Aj3xcyZM0e1tbV6/vnnJUmvvvqqRowYoVmzZmnw4MFauXKlPvjggzpf/tKqVSv95S9/0aFDhxQTE6M777zT9HSWn7v99tuVlZUlb29vPfzww/rd736nrl27auzYsZJOz3MH0PhYHBf7uDoAAHCLmJgY7d271/SqJQAaFlM7AABoBFJSUtSzZ08FBwfrxx9/1PLly5WVlaV33nnH3a0BOA+CNAAAjUBtba2effZZlZaWymKxqEuXLlq2bJlGjBjh7tYAnAdTOwAAAAAT+LAhAAAAYAJBGgAAADCBIA0AAACYwIcNr7BTp06puLhYvr6+9fq6YAAAAFwZDodDR44ckd1uv+AXIhGkr7Di4mIFBwe7uw0AAABcRFFRUZ1vOP05gvQV5uvrK+n0D8bPz8/N3QAAAOBslZWVCg4ONnLb+RCkr7Az0zn8/PwI0gAAAI3Yxabh8mFDAAAAwASCNAAAAGACQRoAAAAwgSANAAAAmECQBgAAAEwgSAMAAAAmEKQBAAAAEwjSAAAAgAkEaQAAAMAEgjQAAABgAkEaAAAAMIEgDQAAAJhAkAYAAABMIEgDAAAAJhCkAQAAABM83d0ATovyjXV3C/WWfWS1u1sAAABwG0akAQAAABMI0gAAAIAJ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LSIiQg899JAKCgqaXYiWpOrqau3cudPpH7imrG/fvnWWnPzmm2900003uamjhrN48WIFBAQoNjbW3a241E8//aRrrnGOOR4eHs1q+bszfHx81K5dO5WXl+ujjz7S/fff7+6WXC4kJERBQUFO2aampkbZ2dluzTaMSDdTKSkpSkhIUEREhCIjI7Vo0SLt27dPjz32mLtbc4mjR4/qP//5j/F6z549Kigo0PXXX6/27du7sTPXGDdunN577z3985//lK+vr/FfF2w2m7y9vd3c3eV76qmndN999yk4OFhHjhxRenq6Pv30U61du9bdrbmEr69vnfnsPj4+at26dbOZ5z5p0iQNHjxY7du3V1lZmV544QVVVlY2m5G+J598Un369FFqaqqGDRumLVu2aNGiRVq0aJG7W3OpU6dOafHixRo5cqQ8PZtXJBg8eLBefPFFtW/fXl27dlV+fr7S0tI0evRod7fmMh999JEcDodCQ0P1n//8R3/4wx8UGhqqUaNGubs1Uy72b3tycrJSU1PVqVMnderUSampqWrVqpXi4+Pd17Tb1gtBg3vttdccN910k8PLy8txxx13NKul0/797387JNXZRo4c6e7WXOJc9ybJsXjxYne35hKjR482/m62bdvW0b9/f0dmZqa722pQzW35u+HDhzvatWvnaNGihcNutzseeOABx/bt293dlkutWrXKERYW5rBarY7bbrvNsWjRIne35HIfffSRQ5Jj165d7m7F5SorKx1PPPGEo3379o6WLVs6br75ZsfTTz/tqK6udndrLvO3v/3NcfPNNzu8vLwcQUFBjnHjxjkOHz7s7rZMu9i/7adOnXJMnz7dERQU5LBarY577rnHsW3bNrf2bHE4HI4rnt4BAACAJo450gAAAIAJBGkAAADABII0AAAAYAJBGgAAADCBIA0AAACYQJAGAAAATCBIAwAAACYQpAEAAAATCNIAgMu2d+9eWSwWFRQUuLsVALhiCNIAcBVJTEyUxWKRxWKRp6en2rdvr8cff1zl5eX1OsfQoUOd9gUHB6ukpERhYWEu7hgAGi+CNABcZQYNGqSSkhLt3btXb7/9tlatWqWxY8de1jk9PDwUFBQkT09PF3UJAI0fQRoArjJWq1VBQUG68cYbFRMTo+HDhyszM1OSVFtbqzFjxigkJETe3t4KDQ3Vq6++arx3xowZWrp0qf75z38aI9uffvppnakdn376qSwWi9atW6eIiAi1atVKffr00a5du5x6eeGFFxQQECBfX1898sgjmjp1qnr06HGlHgUAXBaCNABcxb799lutXbtWLVq0kCSdOnVKN954o/7+979rx44devbZZ/XUU0/p73//uyRp0qRJGjZsmDGqXVJSoj59+pz3/E8//bRefvllbd26VZ6enho9erRxbPny5XrxxRc1Z84c5eXlqX379lq4cGHD3jAAuBD/DQ4ArjL/+te/dO2116q2tlbHjx+XJKWlpUmSWrRooeeee86oDQkJ0caNG/X3v/9dw4YN07XXXitvb29VV1crKCjootd68cUXFRUVJUmaOnWqYmNjdfz4cbVs2VLz58/XmDFjNGrUKEnSs88+q8zMTB09etTVtwwADYIRaQC4ytx7770qKCjQ5s2bNX78eA0cOFDjx483jr/xxhuKiIhQ27Ztde211+qtt97Svn37TF2re/fuxp/btWsnSSorK5Mk7dq1S//zP//jVH/2awBozAjSAHCV8fHx0S233KLu3bvrz3/+s6qrq41R6L///e968sknNXr0aGVmZqqgoECjRo1STU2NqWudmTIiSRaLRdLp6SNn7zvD4XCYug4AuANBGgCuctOnT9e8efNUXFyszz77TH369NHYsWPVs2dP3XLLLfrvf//rVO/l5aXa2trLvm5oaKi2bNnitG/r1q2XfV4AuFII0gBwlevXr5+6du2q1NRU3XLLLdq6das++ugjffPNN3rmmWeUm5vrVN+hQwd99dVX2rVrl3744QedOHHC1HXHjx+vd955R0uXLtXu3bv1wgsv6KuvvqozSg0AjRVBGgCglJQUvfXWWxo6dKgeeOABDR8+XL169dKhQ4fqrDGdlJSk0NBQYx71559/buqaDz30kKZNm6ZJkybpjjvu0J49e5SYmKiWLVu64pYAoMFZHExIAwA0EtHR0QoKCtKyZcvc3QoAXBTL3wEA3OKnn37SG2+8oYEDB8rDw0Pvv/++Pv74Y2VlZbm7NQC4JIxIAwDc4tixYxo8eLC++OILVVdXKzQ0VH/84x/1wAMPuLs1ALgkBGkAAADABD5sCAAAAJhAkAYAAABMIEgDAAAAJhCkAQAAABMI0gAAAIAJBGkAAADABII0AAAAYAJBGgAAADDh/wNJJKtqIp2+EwAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 800x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(8, 5))\n",
    "sns.countplot(data=preprocessed_clean, x='rating', hue='rating', palette='viridis', legend=False)\n",
    "plt.title(\"Distribusi Nilai Rating\")\n",
    "plt.xlabel(\"Rating\")\n",
    "plt.ylabel(\"Jumlah\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d4fe22e5-b17e-4e54-9a38-b270d33fb195",
   "metadata": {},
   "source": [
    "Berdasarkan grafik distribusi nilai rating:\n",
    "* Rating 0 mendominasi jumlah data secara signifikan, dengan lebih dari 600.000 entri. Ini kemungkinan besar menunjukkan rating implisit atau tidak ada rating yang diberikan oleh pengguna.\n",
    "* Rating dari 1 hingga 10 jumlahnya jauh lebih sedikit, dengan puncak pada rating 8, diikuti oleh rating 7, 10, dan 5.\n",
    "* Distribusi rating menunjukkan bahwa ketika pengguna memberikan rating eksplisit, mereka cenderung memberikan nilai tinggi (positif).\n",
    "\n",
    "Kesimpulan:\n",
    "Mayoritas data berisi rating 0 (bisa dianggap sebagai \"tidak diketahui\" atau \"tidak diberikan\"), dan data rating eksplisit menunjukkan kecenderungan bias positif. Hal ini perlu dipertimbangkan dalam pemodelan, misalnya dengan memisahkan rating 0 dari rating eksplisit."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b8cc0c14-c7eb-4fa9-b15d-31f49d4027a7",
   "metadata": {},
   "source": [
    "### 2. Top 10 Buku dengan Rating Terbanyak"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "0df87202-610f-4063-9dbb-cac047eee744",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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K1eu6KOPPtLq1avl6ekpKecRPR9//LECAwM1Y8YMSdLIkSPVtGlTnTlzxhQifvLJJ+rVq5fatGmjDz74QNeuXdPYsWMzhKYP48SJE6pZs6ZKlSqladOmycPDQ5s2bVJ4eLj++OOPDOHGsGHDVKNGDc2dO1dWVlYqWrSoaV+PHj3UqFEjLV26VOfPn9eIESNUr149HTlyxDStLTf38n5t2rRR+/bt1aNHDx09elTDhg2TJNN34c0331TLli21bds2NWzY0HTc119/rfj4eH344YembWfPnlWvXr1UqlQpSdK+ffvUv39//frrrxo1alSm98doNGrw4MH68MMP9emnnyosLOzhbjSA/GUEAADAQ+nWrZvR0dHR9H7u3LlGScaVK1eatXv//feNkoybN282bZNkdHBwMF68eNG0LSUlxRgYGGj08/PLUx3NmjUzent7Z7qvbNmyxtDQ0AzbL1y4YJRknDhxYrZ9jx492igpw8vFxcW4evVqs7ZRUVFGScYzZ86Ybd+xY4dRknHHjh2mbSEhIcYKFSoYU1NTjf369TMWKFDA+L///c/suG7dumV6Xek1Zefq1atGe3t740svvWS2fffu3UZJxpCQENO2SZMmGa2srIwHDhwwa7tq1SqjJOOGDRtM2yQZixUrZkxKSjJtu3jxotHKyso4adIk07ZnnnnG6OXlZbx9+7Zp2/Xr143u7u7Z1p6ammq8e/eucdGiRUZra2vjlStXTPtCQkKMkozfffed2THly5fP9DN+kLe3t7Fp06bGu3fvGu/evWs8d+6csWfPnkZbW1vjunXrsjwuLS3N1F6S8YsvvjDtmzJlSqafeXq999/nM2fOGCUZK1WqZExJSTFt379/v1GScdmyZaZ74OHhYaxevbpZf+fOnTPa2tpm+V3PiiRj3759Te9DQ0ONJUuWNF67ds2sXb9+/Yz29vame57+va1bt26GPtO/61l9vyZMmJBpLdndy/Tv9eTJk82O6dOnj9He3t6YlpZmNBrv3R9fX19jy5Ytzdq98MILxjJlypjaPSj9uzVu3Diju7u7WTtvb29js2bNjDdv3jS2adPG6Orqaty6dWum/QD4d2AaFwAAwGOyfft2OTo6Zhjtkv5fxrdt22a2/fnnn1exYsVM762trdW+fXv9/PPP+uWXXx5bXdlNecrt+iNbt27VgQMHtH//fq1b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lBTduXNHLi4uKl68uHbt2mW2hs6ePXv07LPP5rrP3NxjGxsbNWzYUA0bNtTo0aPl5uam7du3q3Xr1jn2n5KSokWLFmnatGlq3Lix2b42bdpoyZIluVqDycrKSpMmTVLr1q3Vu3dvs33lypXTrl27zKa47dmzx+wzrVmzpry8vLRixQp9/fXXevnll1WgQAFJ974XdnZ2SkhIUEhISI61AAAAAMDjRtiD/4yhQ4fqueeeU9++fdWzZ085OjoqLi5OW7Zs0UcffZTlcampqYqNjTXbVqBAATVs2FCVK1dW586dNWPGDKWkpKhPnz4KCQlRtWrVTG3TA4XevXurfv36ZqM9+vbtq/nz56tjx44aPHiwnnrqKf38889avny55s+fL2tr60xreu2110zhw+7du3N1/ZcvX9bFixeVkpKio0ePaubMmapfv75cXFwkSYMHD9bo0aNVpkwZBQUFKSoqSrGxsRkWmM5OTvd43bp1On36tOrWratChQppw4YNSktLM63Ts2bNGg0bNizLqVzr1q3T1atX1aNHD7OnsElS27ZttWDBglwvuN2sWTNVr15d8+bNU7FixUzbBw8erHbt2ik4OFjPP/+8vvrqK61evVpbt241tTEYDOrUqZPmzp2rU6dOmRZvliRnZ2dFRERo4MCBSktLU+3atZWUlKQ9e/bIyclJ3bp1y/X9BAAAAICHQdiD/4zKlStr586dGj58uOrUqSOj0agyZcqYnp6VlRs3bqhq1apm27y9vXX27FmtXbtW/fv3V926dWVlZaUmTZpkCI5cXFzUokULffbZZxnWlSlevLh2796toUOHKjQ0VLdv35a3t7eaNGkiK6usl9RKX/z48uXLZk/Uyk7Dhg0l3RvR4+npqaZNm+rdd9817Q8PD1dSUpLeeustXbp0SeXLl9eXX36psmXL5qp/Ked77ObmptWrV2vMmDG6deuWypYtq2XLlqlChQqSpGvXrunkyZNZ9r9gwQI1bNgwQ9Aj3RvZM3HiRH3//fcKDg7OVb3vv/++atasabatVatWmjlzpqZMmaLw8HCVLl1aUVFRqlevnlm7zp07a+LEifL29s6wxs/48eNVtGhRTZo0SadPn5abm5uCg4P1zjvv5KouAAAAAHgUBmNunlcM4B/FaDQqMDBQvXr10qBBg/K7HDxhSUlJcnV1VbBnO1lbFcjvcgAA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r4CAAF24cEEbNmxQq1atVK1aNfn4+OjMmTOKjY1VyZIl5ezsrGXLlik1NVXVq1dXwYIFtXjxYjk4OMjb2zvT8wwePFjt2rUzLf781VdfafXq1dq6dWue7q2Pj482bdqkkydPyt3dPdO1fTIzdOhQPffcc+rbt6969uwpR0dHxcXFacuWLfroo4/yVEOZMmWUkpKijz76SC1atNDu3bs1d+7cPPWRzs/PT4sXL1a1atWUlJSkwYMH5zhCq169eurYsaOqVasmd3d3nThxQu+8847q169vNoooXXR0dLaflY+Pj7755ht16NBBdnZ2euqppx7r/XrQ4MGDNXr0aJUpU0ZBQUGKiopSbGysaSpT586dNXr0aHXr1k1jxozR77//rv79++uVV15RsWLFJN1bS+m9995T2bJlVa5cOU2fPl1//vmn2XnWrFmjYcOGZTmVa926dbp69ap69OiR4XvUtm1bLViwIFdhj3RvkfPq1atr3rx5phrTrzWn773BYFCnTp00d+5cnTp1yrR4syQ5OzsrIiJCAwcOVFpammrXrq2kpCTt2bPn/7V359E1Xv3//19HQmQ8xkiipiJDCYIi1NQioTGUqpIiFJ0StNRwt2ooVW21qI8aqonbraLtjVvR1FAxRowpKlIlEe3XXE0MbTRy/f6wcn5OM9KQ9Hg+1jprnXNd+9r7vXf2uqzztq995OLiYvULbAAAAABwLxT7nj0xMTHy9PRUzZo1FRwcrC1btmjOnDn63//+l2uC5XahoaH6/vvv1bp16wIf6ymIl5eXdu7cqZs3byooKEj169fXiBEjZDab89z0uUOHDtq9e7d69+4tb29v9erVS2XLltXmzZtVsWJFSdKQIUP06aefKioqSv7+/mrbtq2ioqIsj6iVKVNGGzZskLu7u7p06SJ/f3+9++67+fbdZDJp/fr1atOmjQYPHixvb289++yzSklJsXxp7dWrl4KDg9W+fXtVrlxZy5cvV7ly5bRo0SK1atVKDRo00ObNm/X1119bYv2rHj16aPbs2Xr//fdVr149LViwQJGRkWrXrt0dje3QoUPl4+Ojpk2bqnLlyvluTHy7Bg0aaOvWrTp+/Lhat26tgIAATZgwocDH6nLTqFEjffjhh5oxY4bq16+vZcuWafr06XdcjyR99tlnunz5sgICAtS/f38NHz5c7u7u+V4TFBSk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uHCh3njjjTtuI7fxkyRXV1dFRUXp6NGjmj17thYtWqSPPvrIcn7dunXq2bOnnnzySR08eFCbN2/O82/1wQcfaPTo0fr222/VsWPHAmPKb6xXrlyphx56SFOmTLH8fbLVqFFD7u7u2r59e551T58+XWaz2fKqVq1agfEAAAAAQDb74g4A99fo0aP15JNPSpImT56sevXq6aeffpKvr6/ef/999evXz7Ihbd26dTVnzhy1bdtWn3zyicqWLZtrnWlpaXJxcZFhGLp+/bqkW6sXfH19JanAelNTU/XNN99o9+7dat68uaRbj0b5+fndVR8L248uXbro5ZdflnRrhchHH32k2NhY+fr6qnLlypKkihUrysPDI8+2Tp06JcMw5OXllePc2rVr5eLiIkm6du2aPD09tXbtWpUqdSvHGh0drVKlSunTTz+1rPyIjIxUuXLlFBsbq6ZNmyotLU0hISGqXbu2JFmNiYuLi+zt7a3iS01NlYeHhzp06KDSpUurevXqatasmSTpxIkTWr58uX7++WdLvKNHj1ZMTIwiIyP1zjvv3OFI39KgQQNNnDhR0q2xnjt3rjZv3qyOHTtqw4YNOnHihGJjYy1xTps2rVDJlILGT5LefPNNy/uaNWtq1KhRWrFihcaMGWNp69lnn9XkyZMt5Ro2bJijrfHjx2vJkiWKjY21So7lJ7+xrlChguzs7OTq6prr/KlatarVCq7c4nnttdcsn9PT00n4AAAAACg0VvY8YBo0aGB57+npKUmWFTT79+9XVFSUXFxcLK+goCBlZWUpOTk5zzpdXV2VkJCg/fv3a/78+apdu7bmz59vOV9QvYmJibK3t7daceHr63vXe8MUth+3j0X2o2h3upro999/l6RcE2Ht27dXQkKCEhISFB8fr06dOqlz5846deqUJc6ffvpJrq6uljgrVKigP/74QydOnFCFChUUFhamoKAgde3aVbNnz7ZaIZKb3r176/fff9fDDz+soUOHatWqVZaVWwcOHJBhGPL29rYam61bt+rEiRN31O/b3T6O0q15lT2OSUlJqlatmlXCIzshUpCCxk+6tbrmsccek4eHh1xcXDRhwgSlpqZazickJOiJJ57It52ZM2dqwYIF2rFjR6ETPVL+Y10QR0dHS2I0Nw4ODnJzc7N6AQAAAEBhsbLnAZP9aJUky2qS7J+BzsrK0gsvvGC1x0u26tWr51lnqVKlVKdOHUm3kjRnz55Vnz59tG3btkLVm723z+37mvwdhe3H7WOR3f6d/iR2pUqVJN16nCt7NVA2Z2dny7hIUpMmTWQ2m7Vo0SJNnTpVWVlZatKkSa77E2XXFRkZqeHDhysmJkYrVqzQm2++qY0bN6pFixa5xlOtWjUlJSVp48aN2rRpk15++WW9//772rp1q7KysmRnZ6f9+/fLzs7O6rrsFTR3I79xNAzjrv+uBY3f7t27Lat2goKCZDabFR0drZkzZ1qucXR0LLCd1q1ba926dfriiy80bty4QseX31j/dUz+6tdff80xXwAAAACgqJDsgUXjxo31ww8/WH3BvhuvvvqqPvzwQ61atUpPPfVUgfX6+fkpMzNT+/bts6z6SEpK0m+//XZX7RdFP8qUKSNJunnzZr7lateuLTc3Nx09elTe3t75ljWZTCpVqpRlNVDjxo21YsUKubu757tyIyAgQAEBARo/frwCAwP1+eefq0WLFipTpkyu8Tk6Oqpbt27q1q2bXnnlFfn6+urw4cMKCAjQzZs3df78ebVu3bqgISgSvr6+Sk1N1blz51SlShVJt35J7W78dfx27typGjVqWO0BdPuqH+nWqqPNmzdr0KBBedbbrFkzRUREKCgoSHZ2dnr99dcLHVNeY924ceM8/z7ZK7cCAgIK3Q4AAAAA3Ake44LF2LFjFRcXp1deeUUJCQk6fvy41qxZo4iIiDuqx83NTUOGDNHEiRNlGEaB9fr4+Cg4OFhDhw5VfHy89u/fryFDhhRqVca96oe7u7scHR0VExOjc+fOKS0tLddypUqVUocOHbRjx44c5zIyMnT27FmdPXtWiYmJioiI0NWrV9W1a1dJUmhoqCpVqqTu3btr+/btSk5O1tatWzVixAj9/PPPSk5O1vjx4xUXF6dTp05pw4YN+vHHHy379tSsWVPJyclKSEjQxYsXlZGRoaioKC1evFhHjhzRyZMntXTpUjk6OqpGjRry9vZWaGioBgwYoJUrVyo5OVl79+7VjBkztH79+rsY6YJ17NhRtWvX1sCBA3Xo0CHt3LnTkpwpaMVPQeNXp04dpaamKjo6WidOnNCcOXO0atUqqzomTpyo5cuXa+LEiUpMTNThw4f13nvv5WgrMDBQ33zzjaZMmWK1wXN+8htr6dbfZ9u2bfrll1908eJFy3W7d++Wg4ODAgMDC9UOAAAAANwpkj2waNCggbZu3arjx4+rdevWCggI0IQJEyx7+9yJESNGKDExUV9++WWh6o2MjFS1atXUtm1b9ezZU8OGDZO7u3ux9cPe3l5z5szRggUL5OXlpe7du+dZdtiwYYqOjs7xCFhMTIw8PT3l6emp5s2ba+/evfryyy8tP+Pu5OSkbdu2qXr16urZs6f8/Pw0ePBg/f7773Jzc5OTk5OOHTumXr16ydvbW8OGDVN4eLheeOEFSVKvXr0UHBys9u3bq3Llylq+fLnKlSunRYsWqVWrVpZVLV9//bXlF6wiIyM1YMAAjRo1Sj4+PurWrZvi4+Pv2ea/dnZ2Wr16ta5evapHH31UQ4YMsWyqnNeG34Udv+7du+vVV19VeHi4GjVqpF27dmnChAlWdbRr105ffvml1qxZo0aNGunxxx9XfHx8ru21atVK69at04QJEzRnzpwC+1bQWE+ZMkUpKSmqXbu21SNby5cvV2hoqJycnApsAwAAAADuhskwDKO4gwD+yQzDUIsWLTRy5Ej17du3uMMp8Xbu3KnHHntMP/30k+VXxh4UFy5ckK+vr/bt26datWoV+rr09HSZzWYlL31Tbk75J8kAFE6Fnm8UXAgAAKCEyf5ukJaWlu92IOzZA/xNJpNJCxcu1KFDh4o7lBJp1apVcnFxUd26dfXTTz9pxIgRatWq1QOX6JGk5ORkzZs3744SPQAAAABwp0j2AEWgYcOGatiwYXGHUSJduXJFY8aM0enTp1WpUiV16NDB6hezHiTNmjUr9E/PAwAAAMDdItkD4J4aMGCABgwYUNxhAAAAAMADgw2aAQAAAAAAbAjJHgAAAAAAABtCsgcAAAAAAMCGkOwBAAAAAACwISR7AAAAAAAAbAjJHgAAAAAAABtCsgcAAAAAAMCGkOwBAAAAAACwIfbFHQAAoHAqdHtdbm5uxR0GAAAAgBKOlT0AAAAAAAA2hGQPAAAAAACADSHZAwAAAAAAYENI9gAAAAAAANgQkj0AAAAAAAA2hGQPAAAAAACADSHZAwAAAAAAYENI9gAAAAAAANgQ++IOAABQOOlHoiUXx+IOAygR3Br0L+4QAAAASixW9gAAAAAAANgQkj0AAAAAAAA2hGQPAAAAAACADSHZAwAAAAAAYENI9gAAAAAAANgQkj0AAAAAAAA2hGQPAAAAAACADSHZAwAAAAAAYENI9gAAAAAAANgQkj0AAAAAAAA2hGQPAAAAAACADSHZAwAAAAAAYENI9gAAAAAAANgQkj0ACpSSkiKTyaSEhITiDgUAAAAAUACSPUAJt2vXLtnZ2Sk4OLi4Q/lb9u7dq1atWsnZ2Vnu7u56+umnlZmZWeB1kyZNkslkkslkkr29vSpVqqQ2bdpo1qxZysjIuA+RAwAAAMA/C8keoIT77LPPFBERoR07dig1NbW4w7lrffr0kaurq/bt26ctW7aoffv2hb62Xr16OnPmjFJTU7Vlyxb17t1b06dPV8uWLXXlypV7GDUAAAAA/POQ7AFKsGvXrumLL77QSy+9pJCQEEVFRVmdj42Nlclk0ubNm9W0aVM5OTmpZcuWSkpKsio3depUubu7y9XVVUOGDNG4cePUqFEjqzKRkZHy8/NT2bJl5evrq3nz5uUZ1+XLlxUaGqrKlSvL0dFRdevWVWRkZL59KVWqlHr27Ck/Pz/Vq1dPr7zyiuzt7Qs1Dvb29vLw8JCXl5f8/f0VERGhrVu36siRI5oxY4al3I0bNzRmzBhVrVpVzs7Oat68uWJjYy3no6KiVK5cOa1du1Y+Pj5ycnLS008/rWvXrmnJkiWqWbOmypcvr4iICN28edOqvwMGDFD58uXl5OSkzp076/jx4znq/fbbb+Xn5ycXFxcFBwfrzJkzljKxsbFq1qyZnJ2dVa5cObVq1UqnTp0qVP8BAAAA4E6Q7AFKsBUrVsjHx0c+Pj567rnnFBkZKcMwcpR74403NHPmTO3bt0/29vYaPHiw5dyyZcs0bdo0zZgxQ/v371f16tX1ySefWF2/aNEivfHGG5o2bZoSExP1zjvvaMKECVqyZEmucU2YMEFHjx7VN998o8TERH3yySeqVKlSvn3p3r27pk6dqpSUlDsfiFz4+vqqc+fOWrlypeXYoEGDtHPnTkVHR+vQoUPq3bu3goODrRIz169f15w5cxQdHa2YmBjFxsaqZ8+eWr9+vdavX6+lS5dq4cKF+uqrryzXhIWFad++fVqzZo3i4uJkGIa6dOmiP//806reDz74QEuXLtW2bduUmpqq0aNHS5IyMzPVo0cPtW3bVocOHVJcXJyGDRsmk8mUa98yMjKUnp5u9QIAAACAwircf6sDKBaLFy/Wc889J0kKDg7W1atXtXnzZnXo0MGq3LRp09S2bVtJ0rhx4/Tkk0/qjz/+UNmyZfXxxx/r+eef16BBgyRJb731ljZs2KCrV69arn/77bc1c+ZM9ezZU5JUq1YtHT16VAsWLNDAgQNzxJWamqqAgAA1bdpUklSzZs18+7FkyRJFRUXp9ddfV9u2bfXNN9/okUcekSR98MEHWrJkiQ4fPnzH4+Pr66sNGzZIkk6cOKHly5fr559/lpeXlyRp9OjRiomJUWRkpN555x1J0p9//qlPPvlEtWvXliQ9/fTTWrp0qc6dOycXFxc98sgjat++vbZs2aI+ffro+PHjWrNmjXbu3KmWLVtKupVAq1atmlavXq3evXtb6p0/f76l3vDwcE2ZMkWSlJ6errS0NIWEhFjO+/n55dmv6dOna/LkyXc8HgAAAAAgsbIHKLGSkpK0Z88ePfvss5JuPcrUp08fffbZZznKNmjQwPLe09NTknT+/HlLPc2aNbMqf/vnCxcu6PTp03r++efl4uJieU2dOlUnTpzINbaXXnpJ0dHRatSokcaMGaNdu3bl2Y+srCyNGzdOb7/9tsaNG6e33npLbdq00e7duyVJR44c0WOPPVaYIcnBMAzL6pgDBw7IMAx5e3tb9WPr1q1W/XBycrIkXCSpSpUqqlmzplxcXKyOZY9fYmKi7O3t1bx5c8v5ihUrysfHR4mJiXnW6+npaamjQoUKCgsLU1BQkLp27arZs2dbPeL1V+PHj1daWprldfr06bsaHwAAAAAPJlb2ACXU4sWLlZmZqapVq1qOGYah0qVL6/LlyypfvrzleOnSpS3vs5MfWVlZOY7dXk+27HKLFi2ySmhIkp2dXa6xde7cWadOndK6deu0adMmPfHEE3rllVf0wQcf5Ch7/vx5nT17VgEBAZKk559/XleuXFGHDh306aef6quvvtJ3332X/2DkITExUbVq1bL0w87OTvv3788R9+2JnNvHSro1Nrkdyx6X3B6byz5++7jmVsft10ZGRmr48OGKiYnRihUr9Oabb2rjxo1q0aJFjrodHBzk4OCQZ78BAAAAID+s7AFKoMzMTP373//WzJkzlZCQYHl9//33qlGjhpYtW1bounx8fLRnzx6rY/v27bO8r1KliqpWraqTJ0+qTp06Vq/sREpuKleurLCwMP3nP//RrFmztHDhwlzLlS9fXo6Ojtq2bZvl2MiRIzVmzBj17dtXTzzxRI6VR4Vx7NgxxcTEqFevXpKkgIAA3bx5U+fPn8/RDw8PjzuuP9sjjzyizMxMxcfHW45dunRJP/74Y76PYuUmICBA48eP165du1S/fn19/vnndx0XAAAAAOSFlT1ACbR27VpdvnxZzz//vMxms9W5p59+WosXL1Z4eHih6oqIiNDQoUPVtGlTtWzZUitWrNChQ4f08MMPW8pMmjRJw4cPl5ubmzp37qyMjAzt27dPly9f1muvvZajzrfeektNmjRRvXr1lJGRobVr1+aZ+HBwcNCIESM0efJkOTk5KTg4WGfPnlVcXJycnZ21fft2JSUlycfHJ88+ZGZm6uzZs8rKytKlS5cUGxurqVOnqlGjRnr99dclSd7e3goNDdWAAQM0c+ZMBQQE6OLFi/ruu+/k7++vLl26FGq8/qpu3brq3r27hg4dqgULFsjV1VXjxo1T1apV1b1790LVkZycrIULF6pbt27y8vJSUlKSfvzxRw0YMOCuYgIAAACA/LCyByiBFi9erA4dOuRI9EhSr169lJCQoAMHDhSqrtDQUI0fP16jR49W48aNlZycrLCwMJUtW9ZSZsiQIfr0008VFRUlf39/tW3bVlFRUXmu7ClTpozGjx+vBg0aqE2bNrKzs1N0dHSeMUybNk0ffvihFi5cqAYNGqhfv37y8fFRSkqKmjVrpieffFIXL17M8/offvhBnp6eql69utq1a6cvvvhC48eP1/bt260e0YqMjNSAAQM0atQo+fj4qFu3boqPj1e1atUKNVZ5iYyMVJMmTRQSEqLAwEAZhqH169fneHQrL05OTjp27Jh69eolb29vDRs2TOHh4XrhhRf+VlwAAAAAkBuTkdeGFABsVseOHeXh4aGlS5cWdygohPT0dJnNZp3euUBuLo7FHQ5QIrg16F/cIQAAANx32d8N0tLS5Obmlmc5HuMCbNz169c1f/58BQUFyc7OTsuXL9emTZu0cePG4g4NAAAAAHAPkOwBbJzJZNL69es1depUZWRkyMfHR//973/VoUOH4g4NAAAAAHAPkOwBbJyjo6M2bdpU3GEAAAAAAO4TNmgGAAAAAACwISR7AAAAAAAAbAjJHgAAAAAAABtCsgcAAAAAAMCGkOwBAAAAAACwISR7AAAAAAAAbAjJHgAAAAAAABtCsgcAAAAAAMCG2Bd3AACAwnGr/6zc3NyKOwwAAAAAJRwrewAAAAAAAGwIyR4AAAAAAAAbQrIHAAAAAADAhrBnDwCUcIZhSJLS09OLORIAAAAAxSn7O0H2d4S8kOwBgBLu0qVLkqRq1aoVcyQAAAAASoIrV67IbDbneZ5kDwCUcBUqVJAkpaam5ntDBwqSnp6uatWq6fTp0/yyG/425hOKEvMJRYW5hKJUEueTYRi6cuWKvLy88i1HsgcASrhSpW5tr2Y2m0vMPzL4Z3Nzc2Muocgwn1CUmE8oKswlFKWSNp8K8x/AbNAMAAAAAABgQ0j2AAAAAAAA2BCSPQBQwjk4OGjixIlycHAo7lDwD8dcQlFiPqEoMZ9QVJhLKEr/5PlkMgr6vS4AAAAAAAD8Y7CyBwAAAAAAwIaQ7AEAAAAAALAhJHsAAAAAAABsCMkeAAAAAAAAG0KyBwBKsHnz5qlWrVoqW7asmjRpou3btxd3SChhJk2aJJPJZPXy8PCwnDcMQ5MmTZKXl5ccHR3Vrl07/fDDD1Z1ZGRkKCIiQpUqVZKzs7O6deumn3/++X53BcVg27Zt6tq1q7y8vGQymbR69Wqr80U1fy5fvqz+/fvLbDbLbDarf//++u233+5x73A/FTSXwsLCctyrWrRoYVWGuQRJmj59uh599FG5urrK3d1dPXr0UFJSklUZ7k0orMLMJ1u9P5HsAYASasWKFRo5cqTeeOMNHTx4UK1bt1bnzp2Vmppa3KGhhKlXr57OnDljeR0+fNhy7r333tOHH36ouXPnau/evfLw8FDHjh115coVS5mRI0dq1apVio6O1o4dO3T16lWFhITo5s2bxdEd3EfXrl1Tw4YNNXfu3FzPF9X86devnxISEhQTE6OYmBglJCSof//+97x/uH8KmkuSFBwcbHWvWr9+vdV55hIkaevWrXrllVe0e/dubdy4UZmZmerUqZOuXbtmKcO9CYVVmPkk2ej9yQAAlEjNmjUzXnzxRatjvr6+xrhx44opIpREEydONBo2bJjruaysLMPDw8N49913Lcf++OMPw2w2G/PnzzcMwzB+++03o3Tp0kZ0dLSlzC+//GKUKlXKiImJuaexo2SRZKxatcryuajmz9GjRw1Jxu7duy1l4uLiDEnGsWPH7nGvUBz+OpcMwzAGDhxodO/ePc9rmEvIy/nz5w1JxtatWw3D4N6Ev+ev88kwbPf+xMoeACiBbty4of3796tTp05Wxzt16qRdu3YVU1QoqY4fPy4vLy/VqlVLzz77rE6ePClJSk5O1tmzZ63mkYODg9q2bWuZR/v379eff/5pVcbLy0v169dnrj3gimr+xMXFyWw2q3nz5pYyLVq0kNlsZo49YGJjY+Xu7i5vb28NHTpU58+ft5xjLiEvaWlpkqQKFSpI4t6Ev+ev8ymbLd6fSPYAQAl08eJF3bx5U1WqVLE6XqVKFZ09e7aYokJJ1Lx5c/373//Wt99+q0WLFuns2bNq2bKlLl26ZJkr+c2js2fPqkyZMipfvnyeZfBgKqr5c/bsWbm7u+eo393dnTn2AOncubOWLVum7777TjNnztTevXv1+OOPKyMjQxJzCbkzDEOvvfaaHnvsMdWvX18S9ybcvdzmk2S79yf7YmkVAFAoJpPJ6rNhGDmO4cHWuXNny3t/f38FBgaqdu3aWrJkiWVzwbuZR8w1ZCuK+ZNbeebYg6VPnz6W9/Xr11fTpk1Vo0YNrVu3Tj179szzOubSgy08PFyHDh3Sjh07cpzj3oQ7ldd8stX7Eyt7AKAEqlSpkuzs7HL8T8D58+dz/E8WcDtnZ2f5+/vr+PHjll/lym8eeXh46MaNG7p8+XKeZfBgKqr54+HhoXPnzuWo/8KFC8yxB5inp6dq1Kih48ePS2IuIaeIiAitWbNGW7Zs0UMPPWQ5zr0JdyOv+ZQbW7k/kewBgBKoTJkyatKkiTZu3Gh1fOPGjWrZsmUxRYV/goyMDCUmJsrT01O1atWSh4eH1Ty6ceOGtm7daplHTZo0UenSpa3KnDlzRkeOHGGuPeCKav4EBgYqLS1Ne/bssZSJj49XWloac+wBdunSJZ0+fVqenp6SmEv4/xmGofDwcK1cuVLfffedatWqZXWeexPuREHzKTc2c3+671tCAwAKJTo62ihdurSxePFi4+jRo8bIkSMNZ2dnIyUlpbhDQwkyatQoIzY21jh58qSxe/duIyQkxHB1dbXMk3fffdcwm83GypUrjcOHDxt9+/Y1PD09jfT0dEsdL774ovHQQw8ZmzZtMg4cOGA8/vjjRsOGDY3MzMzi6hbukytXrhgHDx40Dh48aEgyPvzwQ+PgwYPGqVOnDMMouvkTHBxsNGjQwIiLizPi4uIMf39/IyQk5L73F/dOfnPpypUrxqhRo4xdu3YZycnJxpYtW4zAwECjatWqzCXk8NJLLxlms9mIjY01zpw5Y3ldv37dUoZ7EwqroPlky/cnkj0AUIL93//9n1GjRg2jTJkyRuPGja1+JhIwDMPo06eP4enpaZQuXdrw8vIyevbsafzwww+W81lZWcbEiRMNDw8Pw8HBwWjTpo1x+PBhqzp+//13Izw83KhQoYLh6OhohISEGKmpqfe7KygGW7ZsMSTleA0cONAwjKKbP5cuXTJCQ0MNV1dXw9XV1QgNDTUuX758n3qJ+yG/uXT9+nWjU6dORuXKlY3SpUsb1atXNwYOHJhjnjCXYBhGrvNIkhEZGWkpw70JhVXQfLLl+5PJMAzj/q0jAgAAAAAAwL3Enj0AAAAAAAA2hGQPAAAAAACADSHZAwAAAAAAYENI9gAAAAAAANgQkj0AAAAAAAA2hGQPAAAAAACADSHZAwAAAAAAYENI9gAAAAAAANgQkj0AAABAEZo0aZIaNWp0R9eYTCatXr36nsRzN2JjY2UymfTbb78VdygAgLtAsgcAAAAPjLCwMPXo0aO4wygS7dq1k8lkkslkUpkyZVS7dm2NHz9eGRkZd1zPyJEjrY61bNlSZ86ckdlsLsKIAQD3i31xBwAAAADg7gwdOlRTpkzRjRs3tHfvXg0aNEiSNH369L9Vb5kyZeTh4VEUIQIAigErewAAAPBAqlmzpmbNmmV1rFGjRpo0aZLls8lk0oIFCxQSEiInJyf5+fkpLi5OP/30k9q1aydnZ2cFBgbqxIkTebazd+9edezYUZUqVZLZbFbbtm114MCBHOUuXryop556Sk5OTqpbt67WrFlTYB+cnJzk4eGh6tWrq1evXurYsaM2bNhgOX/p0iX17dtXDz30kJycnOTv76/ly5dbzoeFhWnr1q2aPXu2ZZVQSkpKjse4oqKiVK5cOX377bfy8/OTi4uLgoODdebMGUtdmZmZGj58uMqVK6eKFStq7NixGjhwoM2spAKAfxKSPQAAAEA+3n77bQ0YMEAJCQny9fVVv3799MILL2j8+PHat2+fJCk8PDzP669cuaKBAwdq+/bt2r17t+rWrasuXbroypUrVuUmT56sZ555RocOHVKXLl0UGhqqX3/9tdBxfv/999q5c6dKly5tOfbHH3+oSZMmWrt2rY4cOaJhw4apf//+io+PlyTNnj1bgYGBGjp0qM6cOaMzZ86oWrVqudZ//fp1ffDBB1q6dKm2bdum1NRUjR492nJ+xowZWrZsmSIjI7Vz506lp6eXqH2IAOBBQrIHAAAAyMegQYP0zDPPyNvbW2PHjlVKSopCQ0MVFBQkPz8/jRgxQrGxsXle//jjj+u5556Tn5+f/Pz8tGDBAl2/fl1bt261KhcWFqa+ffuqTp06euedd3Tt2jXt2bMn39jmzZsnFxcXOTg4qFGjRrpw4YJef/11y/mqVatq9OjRatSokR5++GFFREQoKChIX375pSTJbDarTJkylhVCHh4esrOzy7WtP//8U/Pnz1fTpk3VuHFjhYeHa/PmzZbzH3/8scaPH6+nnnpKvr6+mjt3rsqVK1fA6AIA7gWSPQAAAEA+GjRoYHlfpUoVSZK/v7/VsT/++EPp6em5Xn/+/Hm9+OKL8vb2ltlsltls1tWrV5WamppnO87OznJ1ddX58+fzjS00NFQJCQmKi4vTM888o8GDB6tXr16W8zdv3tS0adPUoEEDVaxYUS4uLtqwYUOOtgvDyclJtWvXtnz29PS0xJeWlqZz586pWbNmlvN2dnZq0qTJHbcDAPj72KAZAAAAD6RSpUrJMAyrY3/++WeOcrc/FmUymfI8lpWVlWs7YWFhunDhgmbNmqUaNWrIwcFBgYGBunHjRp7tZNebV53ZzGaz6tSpI0n6z3/+o3r16mnx4sV6/vnnJUkzZ87URx99pFmzZsnf31/Ozs4aOXJkjrYLI7f4/jp+2WOR7a/nAQD3Byt7AAAA8ECqXLmy1QbD6enpSk5OLvJ2tm/fruHDh6tLly6qV6+eHBwcdPHixSJvp3Tp0vrXv/6lN998U9evX7e03b17dz333HNq2LChHn74YR0/ftzqujJlyujmzZt/q22z2awqVapYPXZ28+ZNHTx48G/VCwC4OyR7AAAA8EB6/PHHtXTpUm3fvl1HjhzRwIED89yv5u+oU6eOli5dqsTERMXHxys0NFSOjo5F3o4k9evXTyaTSfPmzbO0vXHjRu3atUuJiYl64YUXdPbsWatratasqfj4eKWkpOjixYsFribKS0REhKZPn67//e9/SkpK0ogRI3T58uUcq30AAPceyR4AAAA8MLKysmRvf2sng/Hjx6tNmzYKCQlRly5d1KNHD6s9aYrKZ599psuXLysgIED9+/fX8OHD5e7uXuTtSLdW6YSHh+u9997T1atXNWHCBDVu3FhBQUFq166dPDw8cvwU+ujRo2VnZ6dHHnlElStXvqv9fCRp7Nix6tu3rwYMGKDAwEC5uLgoKChIZcuWLYKeAQDuhMngQVoAAAA8IIKDg1WnTh3NnTu3uEOxeVlZWfLz89Mzzzyjt99+u7jDAYAHChs0AwAAwOZdvnxZu3btUmxsrF588cXiDscmnTp1Shs2bFDbtm2VkZGhuXPnKjk5Wf369Svu0ADggUOyBwAAADZv8ODB2rt3r0aNGqXu3bsXdzg2qVSpUoqKitLo0aNlGIbq16+vTZs2yc/Pr7hDA4AHDo9xAQAAAAAA2BA2aAYAAAAAALAhJHsAAAAAAABsCMkeAAAAAAAAG0KyBwAAAAAAwIaQ7AEAAAAAALAhJHsAAAAAAABsCMkeAAAAAAAAG0KyBwAAAAAAwIb8f0d4WUZ1bwxCAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "top_books = preprocessed_clean['book_title'].value_counts().head(10)\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.barplot(x=top_books.values, y=top_books.index, hue=top_books.index, palette='magma', legend=False)\n",
    "plt.title(\"Top 10 Buku dengan Rating Terbanyak\")\n",
    "plt.xlabel(\"Jumlah Rating\")\n",
    "plt.ylabel(\"Judul Buku\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a91eeb34-3883-4b96-a06c-8788a3f39dfc",
   "metadata": {},
   "source": [
    "Berdasarkan grafik Top 10 Buku dengan Rating Terbanyak:\n",
    "* Buku \"Wild Animus\" memperoleh jumlah rating terbanyak secara signifikan dibandingkan buku lain, melebihi 2.500 rating.\n",
    "* Buku populer lain seperti \"The Lovely Bones: A Novel\", \"The Da Vinci Code\", dan \"The Secret Life of Bees\" juga masuk dalam daftar dengan jumlah rating yang tinggi.\n",
    "* Grafik ini mencerminkan buku-buku yang paling sering dinilai oleh pengguna, bukan buku dengan rating tertinggi secara kualitas.\n",
    "\n",
    "Kesimpulan:\n",
    "Buku-buku pada grafik ini bisa dianggap sebagai buku populer atau paling dikenal oleh komunitas pengguna, sehingga cocok dijadikan acuan awal untuk sistem rekomendasi berbasis popularitas atau cold-start (tanpa data pengguna)."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b4ba6001-59ee-45d8-80a6-41be933e13fe",
   "metadata": {},
   "source": [
    "### 3. Distribusi Usia Pengguna"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "b0dd7f29-1903-43c9-94d2-a4deff45e315",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 800x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(8, 5))\n",
    "sns.histplot(preprocessed_clean['age'], bins=30, kde=True, color='skyblue')\n",
    "plt.title(\"Distribusi Usia Pengguna\")\n",
    "plt.xlabel(\"Usia\")\n",
    "plt.ylabel(\"Jumlah\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e2d3aab9-f73d-4a53-9a98-58ba6030d24a",
   "metadata": {},
   "source": [
    "Berdasarkan grafik Distribusi Usia Pengguna:\n",
    "* Mayoritas pengguna berada pada rentang usia 30–40 tahun, dengan puncak tertinggi sekitar usia 35 tahun.\n",
    "* Terlihat adanya penurunan tajam di luar rentang usia tersebut, terutama setelah usia 60 tahun.\n",
    "* Distribusi menunjukkan pola right-skewed, artinya lebih banyak pengguna muda hingga paruh baya dibandingkan pengguna lansia.\n",
    "\n",
    "Kesimpulan:\n",
    "Sistem rekomendasi buku sebaiknya mempertimbangkan dominasi usia 30–40 tahun ini, karena preferensi mereka kemungkinan besar paling merepresentasikan tren umum dalam data."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c4f52dcd-75b1-42da-bb58-c05cc76b9b3f",
   "metadata": {},
   "source": [
    "### 4. Top 10 Bahasa Buku Terpopuler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "68e2fa0a-29c9-40e6-9c8b-34265cfcba87",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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gwcPdv5LfteuXTVnzhz17dtX99xzj5KSkjRq1CiFh4dr586defsQL9G1a1d98cUXqlGjhurVq6f169frjTfeyBEib7vtNtWpU0eNGzdWaGio9u/fr/HjxysqKkqxsbF/q68rOXXqlNasWSPpwmpuv//+u0aPHi273a5+/fo52z300EOaNGmSevTooccff1zHjx/X2LFjnYEyN+PGjVNgYKD69u2r1NRUDR061Nnfyy+/rFdeeUVxcXHavn273n//feeY5FdAQIDee+899erVSydOnNA999yj8uXL6+jRo9q0aZOOHj2aI7DlxtfXV+PGjVNqaqpuuukmrVq1Sq+++qo6deqkVq1aSZJuueUWdezYUc8//7wcDodatmypzZs3a/jw4WrYsKEeeughlz4jIiJ0++23KyEhQeHh4Zo6daoWLlyo119/3fl8optuuknVq1fXs88+q8zMTJUpU0Zz587VihUrrlpzy5Yt9cQTT+jhhx/Wr7/+qptvvln+/v5KTk7WihUrVLduXT311FN/81MFgEu4d90IAChcl65uZ4wxx48fN3369DHh4eHG29vbREVFmWHDhplz5865tJNk+vXrZyZMmGCqVq1qfHx8TI0aNcxXX32Vp/fObdU5XWYVtNOnT5sBAwaYsLAw4+vra+rVq2emTZuW7/cJCQkxTZs2NZ999pnJyspyaT9mzBgTHR1t7Ha7qVmzpvn444+dfVzu+i916WptJ0+eNI8++qgpX768KVWqlGnVqpX5+eefc6weN27cONOiRQtTrlw54+vra2644Qbz6KOPuqwSl9e+cnPp6nYlSpQwN9xwg7nnnnvMb7/9lqP95MmTTc2aNU3JkiVNrVq1zIwZM664up3VG2+84bKSXHp6unnuuedMZGSk8fPzM3FxcWbjxo3/eHW7i5YtW2a6dOliQkJCjI+Pj6lYsaLp0qWLS7uL43j06NEc/V78Xdi8ebOJj483fn5+JiQkxDz11FMmNTXVpe3Zs2fN888/b6KiooyPj48JDw83Tz31lDl58qRLu6ioKNOlSxcza9YsU7t2bePr62uio6PNW2+9leP9//zzT9OhQwcTFBRkQkNDzdNPP23mz59/1dXtLvrss89M06ZNjb+/v/Hz8zNVq1Y1PXv2NL/++quzTVxcnKldu/aVPl4AuCybMcYUaioDgGLKZrOpX79+ev/9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",
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "top_lang = preprocessed_clean['Language'].value_counts().head(10)\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.barplot(x=top_lang.values, y=top_lang.index, hue=top_lang.index, palette='coolwarm', legend=False)\n",
    "plt.title(\"Top 10 Bahasa Buku Terpopuler\")\n",
    "plt.xlabel(\"Jumlah Buku\")\n",
    "plt.ylabel(\"Bahasa\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9f2bbd4a-f903-4ca0-8e5d-e1b536c4c96a",
   "metadata": {},
   "source": [
    "Grafik Top 10 Bahasa Buku Terpopuler menunjukkan distribusi bahasa dari buku-buku yang tersedia dalam dataset:\n",
    "\n",
    "* Bahasa Inggris (en) mendominasi secara signifikan dengan jumlah buku terbanyak (lebih dari 600.000 judul).\n",
    "* Label ‘9’ muncul sebagai bahasa kedua terbanyak, yang kemungkinan besar merupakan data anomali atau kode yang salah dalam pengisian kolom bahasa.\n",
    "* Bahasa lainnya seperti Jerman (de), Spanyol (es), Prancis (fr), dan beberapa bahasa Eropa lainnya muncul dalam jumlah yang jauh lebih kecil.\n",
    "\n",
    "Kesimpulan:\n",
    "Mayoritas buku dalam dataset berbahasa Inggris, sehingga sistem rekomendasi kemungkinan akan lebih relevan bagi pengguna yang memahami bahasa tersebut. Perlu penanganan khusus terhadap entri bahasa tidak valid seperti ‘9’."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cb446db4-e817-47a1-8fad-020f818fc690",
   "metadata": {},
   "source": [
    "### 5. Top 10 Kategori Buku"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "e3c3ea16-e0af-486c-a0c6-5f284d238f20",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABAgAAAIhCAYAAADD6ojeAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAdH5JREFUeJzt3XlYFeX///HXYTuyiwuChpK7qGiJlkvu5pKmablkKlqWuadm+bESM8NM07RP2qLirpVL5pZogruihkviRhr2kSI3cEWU+f3hj/P1CCgicEifj+s6V5yZe+55z32mrs7r3DNjMgzDEAAAAAAAeKTZ2boAAAAAAABgewQEAAAAAACAgAAAAAAAABAQAAAAAAAAERAAAAAAAAAREAAAAAAAABEQAAAAAAAAERAAAAAAAAAREAAAAAAAABEQAACAB2AymbL0ioiIyPVa5syZo86dO6tChQqys7OTv79/pm0vXbqkwYMHq3jx4ipQoICqV6+uRYsWZWk/ISEhMplMOnPmjNXy2NhYlS5dWsWKFVN0dPR91b5gwQJNnjz5vraxpYiIiCx9rmntbn95eXnpqaee0uzZs7O9/8w+AwDAg3GwdQEAAODfa/v27Vbvx4wZo40bN+qXX36xWh4QEJDrtcydO1d//fWXatWqpdTUVKWkpGTatn379oqKitK4ceNUvnx5LViwQF26dFFqaqpefvnl+973gQMH1Lx5czk6OmrLli0qV67cfW2/YMECHTx4UIMHD77vfdvCk08+qe3bt2f5c/3444/VqFEjSdKZM2c0Z84cBQcHKykpSQMGDMjNUgEA94GAAAAAZNvTTz9t9b5o0aKys7NLtzwv/Pzzz7KzuzU5snXr1jp48GCG7VavXq3w8HBLKCBJjRo10h9//KG3335bnTp1kr29fZb3u2PHDrVq1UrFihVTeHi4HnvssQc/mHwqJSVFJpNJHh4e9/UZlytXzqp9q1atFBUVpYULFxIQAEA+wiUGAAAgV507d059+/ZViRIl5OTkpNKlS2vkyJFKTk62amcymdS/f3999dVXKl++vMxmswICArI89T8tHLiXZcuWyc3NTS+99JLV8p49e+r06dPauXNn1g5MUnh4uJo2baoyZcpo8+bN6cKB//73v6pfv768vb3l6uqqqlWravz48VazGxo2bKhVq1bpjz/+sJqKn+b69ev6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nlVkZGRO1whY3P4EgtTUVLVp00affPJJuna+vr737MswDKsv+lmR0TaGYUiS1XJHR0erNiaTSampqZn2azabZTab76sWAAAAAMhJ9xUQvPjii7pw4YJefvllrVq1So0aNdKlS5fUokULnTlzRpGRkdykEHnmySef1JIlS+Tv75+tpwxUrFhRcXFx+vvvv1WsWDFJtx6DeDcBAQFasmSJVVCwbds2ubu7q0SJEvd/EAAAAACQT9z3TQpfe+01hYSEqF27doqIiFDLli31119/aePGjVn61RbIKf369dO5c+fUpUsX7dq1S7///rvWrVunXr166ebNm/fcvlmzZipTpox69Oih/fv3a+vWrZabFGY2s6Bv3746deqUBgwYoMOHD+vHH3/UqFGjNGTIEMv9BwAAAADg3yhb32iGDx+uvn37qkmTJjp9+rQiIiL49RR5rnjx4tq6datu3ryp5s2bq0qVKho0aJA8PT2z9GXd3t5ey5cv16VLl1SzZk299tpreu+99yRJBQoUyHCbEiVKaPXq1dq1a5eqVaumPn366NVXX7VsBwAAAAD/ViYj7QLqLGjfvr3V+9WrV6tatWrpwoGlS5fmTHVAHtu6davq1aun48ePq0yZMjarIykpSZ6enjq8oanc3RzvvcFtitdanUtVAQAAAMhrad8NEhMT5eHhkav7uq8Ltz09Pa3ed+nSJUeLAfLasmXL5ObmpnLlyun48eMaNGiQ6tata9NwAAAAAABs4b4CgowecQj8m128eFHDhw/XqVOnVKRIETVt2lQTJ060dVkAAAAAkOfu/9bvwEOke/fu6t69u63LAAAAAACb47brAAAAAACAgAAAAAAAABAQAAAAAAAAERAAAAAAAAAREAAAAAAAABEQAAAAAAAAERAAAAAAAAAREAAAAAAAAEkOti4AQOZ8g5bIw8PD1mUAAAAAeAQwgwAAAAAAABAQAAAAAAAAAgIAAAAAACACAgAAAAAAIAICAAAAAAAgAgIAAAAAACACAgAAAAAAIAICAAAAAAAgAgIAAAAAACDJwdYFAMhc7LZ5cnd1vmubss/0zKNqAAAAADzMmEEAAAAAAAAICAAAAAAAAAEBAAAAAAAQAQEAAAAAABABAQAAAAAAEAEBAAAAAAAQAQEAAAAAABABAQAAAAAAEAEBAAAAAAAQAQEAAAAAABABAQAAAAAA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uDirX9APHTqkxMREVapUKUtjlTYmW7ZssSxLSUnR7t27s9xHVlStWlWpqamKjIzMtE3RokUVHBysefPmafLkyfr6669zbP8AAAAAkJMICDJx6tQpDRkyREeOHNHChQs1depUDRo0yLK+cePG+uKLL7R3717t3r1bffr0sfr1edKkSVq0aJEOHz6so0eP6vvvv5ePj48KFiyYrXreeecdbd++Xf369VN0dLSOHTumFStWaMCAAZKklStXasqUKYqOjtYff/yhOXPmKDU1VRUqVNDOnTv18ccfa/fu3YqLi9PSpUv1zz//ZPpl+dVXX9XNmzf1/PPPa9u2bTpy5IhWrFihCxcuyMXF5a43a+zatauKFCmitm3bavPmzTpx4oQiIyM1aNAg/fnnn5Ju3Rhw4sSJmjJlio4dO6a9e/dabgTYtGlTBQYGqmvXrtq7d6927dql7t27q0GDBgoKCsrSWLm6uurNN9/U22+/rbVr1+rQoUPq3bu3rly5oldfffV+hv2u/P391aNHD/Xq1UvLly/XiRMnFBERoe+++06S9MEHH+jHH3/U8ePH9dtvv2nlypU5GlAAAAAAQE5ysHUB+VX37t119epV1apVS/b29howYIBef/11y/qJEyeqZ8+eql+/vooXL67PP/9ce/bssax3c3PTJ598omPHjsne3l41a9bU6tWrZWeXvUwmMDBQkZGRGjlypJ555hkZhqEyZcpY7u5fsGBBLV26VCEhIbp27ZrKlSunhQsXqnLlyoqJidGmTZs0efJkJSUlqVSpUpo4caJatmyZ4b6KFy+uXbt26Z133lH79u2VlJSkGjVqaMGCBXJxcVGzZs1UtmxZDRkyJN22Li4u2rRpk2XbixcvqkSJEmrSpIll2nyPHj107do1TZo0ScOGDVORIkUsl1+kPf5xwIABql+/vuzs7NSiRYu7PkkgI+PGjVNqaqq6deumixcvKigoSD///LO8vLzuq597mTZtmv7zn/+ob9++Onv2rEqWLKn//Oc/kiQnJyeNGDFCJ0+elLOzs5555hktWrQoR/cPAAAAADnFZNx5IT0Am0tKSpKnp6eOTxwod2dzpu283xyfh1UBAAAAyGtp3w0SExNz/V5lXGIAAAAAAAAICAAAAAAAAAEBAAAAAAAQAQEAAAAAABABAQAAAAAAEAEBAAAAAAAQAQEAAAAAABAB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",
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "all_categories = [cat.strip().lower() for sublist in preprocessed_clean['Category'] for cat in sublist]\n",
    "cat_series = pd.Series(all_categories)\n",
    "top_categories = cat_series.value_counts().head(10)\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.barplot(\n",
    "    x=top_categories.values,\n",
    "    y=top_categories.index,\n",
    "    hue=top_categories.index,\n",
    "    palette='Set2',\n",
    "    legend=False\n",
    ")\n",
    "plt.title(\"Top 10 Kategori Buku\")\n",
    "plt.xlabel(\"Jumlah\")\n",
    "plt.ylabel(\"Kategori\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "523aa0e7-0ebf-40ea-a816-89caad84918f",
   "metadata": {},
   "source": [
    "Grafik Top 10 Kategori Buku menunjukkan distribusi kategori buku yang paling banyak terdapat dalam dataset:\n",
    "\n",
    "* Kategori \"Fiction\" mendominasi secara signifikan, dengan jumlah buku hampir 400.000 judul. Ini menunjukkan bahwa fiksi adalah genre yang paling umum dan populer dalam koleksi data ini.\n",
    "*Diikuti oleh \"Juvenile Fiction\" (fiksi untuk remaja) dan \"Biography & Autobiography\", yang juga memiliki jumlah signifikan namun jauh lebih kecil dibandingkan fiksi umum.\n",
    "* Kategori lain seperti \"Humor\", \"History\", dan \"Religion\" muncul dalam jumlah lebih terbatas.\n",
    "\n",
    "Kesimpulan: Genre fiksi mendominasi isi dataset, sehingga sistem rekomendasi kemungkinan akan lebih banyak merekomendasikan buku-buku dalam kategori ini kecuali dilakukan penyesuaian khusus untuk genre lainnya."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d137a920-a621-4da6-b5aa-e72f23cc8a30",
   "metadata": {},
   "source": [
    "# **Modelling**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "71b985ca-6527-45d0-b338-89456685cff0",
   "metadata": {},
   "source": [
    "# **Content-Based Filtering**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d778846d-0bf4-4823-b29a-d3855b7ba11f",
   "metadata": {},
   "source": [
    "### A. Data Preparation"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b33a6e20-796f-4054-9ab2-1effc708eebf",
   "metadata": {},
   "source": [
    "Dalam pendekatan Content-Based Filtering, sistem rekomendasi akan memberikan saran buku yang mirip secara konten dengan buku yang pernah disukai pengguna. Fokus utama metode ini adalah menganalisis informasi deskriptif dari setiap item (dalam hal ini, buku) untuk menemukan kemiripan antar buku berdasarkan atribut-atribut berikut:\n",
    "\n",
    "📖 Summary: Ringkasan isi buku yang menggambarkan topik atau cerita.\n",
    "\n",
    "🏷️ Category: Kategori atau genre buku, misalnya [Fiction], [Science], [Biography], dll.\n",
    "\n",
    "📘 book_title: Judul buku, karena bisa mengandung kata-kata penting terkait isi.\n",
    "\n",
    "🏢 publisher: Nama penerbit, yang dalam beberapa kasus mengindikasikan genre atau kualitas buku.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "4b3c8cfe-ec80-444f-a224-20a621e6b3b9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<bound method NDFrame.head of          user_id                   location      age        isbn  rating  \\\n",
       "0              2  stockton, california, usa  18.0000  0195153448       0   \n",
       "1              8   timmins, ontario, canada  34.7439  0002005018       5   \n",
       "15             8   timmins, ontario, canada  34.7439  0060973129       0   \n",
       "18             8   timmins, ontario, canada  34.7439  0374157065       0   \n",
       "29             8   timmins, ontario, canada  34.7439  0393045218       0   \n",
       "...          ...                        ...      ...         ...     ...   \n",
       "1031170   278851         dallas, texas, usa  33.0000  0743203763       0   \n",
       "1031171   278851         dallas, texas, usa  33.0000  0767907566       5   \n",
       "1031172   278851         dallas, texas, usa  33.0000  0884159221       7   \n",
       "1031173   278851         dallas, texas, usa  33.0000  0912333022       7   \n",
       "1031174   278851         dallas, texas, usa  33.0000  1569661057      10   \n",
       "\n",
       "                                                book_title  \\\n",
       "0                                      Classical Mythology   \n",
       "1                                             Clara Callan   \n",
       "15                                    Decision in Normandy   \n",
       "18       Flu: The Story of the Great Influenza Pandemic...   \n",
       "29                                  The Mummies of Urumchi   \n",
       "...                                                    ...   \n",
       "1031170  As Hogan Said . . . : The 389 Best Things Anyo...   \n",
       "1031171  All Elevations Unknown: An Adventure in the He...   \n",
       "1031172  Why stop?: A guide to Texas historical roadsid...   \n",
       "1031173  The Are You Being Served? Stories: 'Camping In...   \n",
       "1031174  Dallas Street Map Guide and Directory, 2000 Ed...   \n",
       "\n",
       "                  book_author  year_of_publication                 publisher  \\\n",
       "0          Mark P. O. Morford               2002.0   Oxford University Press   \n",
       "1        Richard Bruce Wright               2001.0     HarperFlamingo Canada   \n",
       "15               Carlo D'Este               1991.0           HarperPerennial   \n",
       "18           Gina Bari Kolata               1999.0      Farrar Straus Giroux   \n",
       "29            E. J. W. Barber               1999.0    W. W. Norton & Company   \n",
       "...                       ...                  ...                       ...   \n",
       "1031170        Randy Voorhees               2000.0          Simon & Schuster   \n",
       "1031171          Sam Lightner               2001.0            Broadway Books   \n",
       "1031172         Claude Dooley               1985.0           Lone Star Books   \n",
       "1031173          Jeremy Lloyd               1997.0                Kqed Books   \n",
       "1031174                Mapsco               1999.0  American Map Corporation   \n",
       "\n",
       "                                                     img_s  \\\n",
       "0        http://images.amazon.com/images/P/0195153448.0...   \n",
       "1        http://images.amazon.com/images/P/0002005018.0...   \n",
       "15       http://images.amazon.com/images/P/0060973129.0...   \n",
       "18       http://images.amazon.com/images/P/0374157065.0...   \n",
       "29       http://images.amazon.com/images/P/0393045218.0...   \n",
       "...                                                    ...   \n",
       "1031170  http://images.amazon.com/images/P/0743203763.0...   \n",
       "1031171  http://images.amazon.com/images/P/0767907566.0...   \n",
       "1031172  http://images.amazon.com/images/P/0884159221.0...   \n",
       "1031173  http://images.amazon.com/images/P/0912333022.0...   \n",
       "1031174  http://images.amazon.com/images/P/1569661057.0...   \n",
       "\n",
       "                                                     img_m  \\\n",
       "0        http://images.amazon.com/images/P/0195153448.0...   \n",
       "1        http://images.amazon.com/images/P/0002005018.0...   \n",
       "15       http://images.amazon.com/images/P/0060973129.0...   \n",
       "18       http://images.amazon.com/images/P/0374157065.0...   \n",
       "29       http://images.amazon.com/images/P/0393045218.0...   \n",
       "...                                                    ...   \n",
       "1031170  http://images.amazon.com/images/P/0743203763.0...   \n",
       "1031171  http://images.amazon.com/images/P/0767907566.0...   \n",
       "1031172  http://images.amazon.com/images/P/0884159221.0...   \n",
       "1031173  http://images.amazon.com/images/P/0912333022.0...   \n",
       "1031174  http://images.amazon.com/images/P/1569661057.0...   \n",
       "\n",
       "                                                     img_l  \\\n",
       "0        http://images.amazon.com/images/P/0195153448.0...   \n",
       "1        http://images.amazon.com/images/P/0002005018.0...   \n",
       "15       http://images.amazon.com/images/P/0060973129.0...   \n",
       "18       http://images.amazon.com/images/P/0374157065.0...   \n",
       "29       http://images.amazon.com/images/P/0393045218.0...   \n",
       "...                                                    ...   \n",
       "1031170  http://images.amazon.com/images/P/0743203763.0...   \n",
       "1031171  http://images.amazon.com/images/P/0767907566.0...   \n",
       "1031172  http://images.amazon.com/images/P/0884159221.0...   \n",
       "1031173  http://images.amazon.com/images/P/0912333022.0...   \n",
       "1031174  http://images.amazon.com/images/P/1569661057.0...   \n",
       "\n",
       "                                                   Summary Language  \\\n",
       "0        Provides an introduction to classical myths pl...       en   \n",
       "1        In a small town in Canada, Clara Callan reluct...       en   \n",
       "15       Here, for the first time in paperback, is an o...       en   \n",
       "18       Describes the great flu epidemic of 1918, an o...       en   \n",
       "29       A look at the incredibly well-preserved ancien...       en   \n",
       "...                                                    ...      ...   \n",
       "1031170  Golf lovers will revel in this collection of t...       en   \n",
       "1031171  A daring twist on the travel-adventure genre t...       en   \n",
       "1031172                                                  9        9   \n",
       "1031173  These hilarious stories by the creator of publ...       en   \n",
       "1031174                                                  9        9   \n",
       "\n",
       "                 Category      city       state country  \\\n",
       "0        [Social Science]  stockton  california     usa   \n",
       "1             [Actresses]   timmins     ontario  canada   \n",
       "15            [1940-1949]   timmins     ontario  canada   \n",
       "18              [Medical]   timmins     ontario  canada   \n",
       "29               [Design]   timmins     ontario  canada   \n",
       "...                   ...       ...         ...     ...   \n",
       "1031170           [Humor]    dallas       texas     usa   \n",
       "1031171          [Nature]    dallas       texas     usa   \n",
       "1031172                []    dallas       texas     usa   \n",
       "1031173         [Fiction]    dallas       texas     usa   \n",
       "1031174                []    dallas       texas     usa   \n",
       "\n",
       "                                                  combined  \n",
       "0        Provides an introduction to classical myths pl...  \n",
       "1        In a small town in Canada, Clara Callan reluct...  \n",
       "15       Here, for the first time in paperback, is an o...  \n",
       "18       Describes the great flu epidemic of 1918, an o...  \n",
       "29       A look at the incredibly well-preserved ancien...  \n",
       "...                                                    ...  \n",
       "1031170  Golf lovers will revel in this collection of t...  \n",
       "1031171  A daring twist on the travel-adventure genre t...  \n",
       "1031172  9 [] Why stop?: A guide to Texas historical ro...  \n",
       "1031173  These hilarious stories by the creator of publ...  \n",
       "1031174  9 [] Dallas Street Map Guide and Directory, 20...  \n",
       "\n",
       "[267330 rows x 19 columns]>"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "preprocessed_clean.head"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "e3740a7a-57ca-4140-9868-ff177e6f1a40",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>user_id</th>\n",
       "      <th>location</th>\n",
       "      <th>age</th>\n",
       "      <th>isbn</th>\n",
       "      <th>rating</th>\n",
       "      <th>book_title</th>\n",
       "      <th>book_author</th>\n",
       "      <th>year_of_publication</th>\n",
       "      <th>publisher</th>\n",
       "      <th>img_s</th>\n",
       "      <th>img_m</th>\n",
       "      <th>img_l</th>\n",
       "      <th>Summary</th>\n",
       "      <th>Language</th>\n",
       "      <th>Category</th>\n",
       "      <th>city</th>\n",
       "      <th>state</th>\n",
       "      <th>country</th>\n",
       "      <th>combined</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2</td>\n",
       "      <td>stockton, california, usa</td>\n",
       "      <td>18.0000</td>\n",
       "      <td>0195153448</td>\n",
       "      <td>0</td>\n",
       "      <td>Classical Mythology</td>\n",
       "      <td>Mark P. O. Morford</td>\n",
       "      <td>2002.0</td>\n",
       "      <td>Oxford University Press</td>\n",
       "      <td>http://images.amazon.com/images/P/0195153448.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0195153448.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0195153448.0...</td>\n",
       "      <td>Provides an introduction to classical myths pl...</td>\n",
       "      <td>en</td>\n",
       "      <td>[Social Science]</td>\n",
       "      <td>stockton</td>\n",
       "      <td>california</td>\n",
       "      <td>usa</td>\n",
       "      <td>Provides an introduction to classical myths pl...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>8</td>\n",
       "      <td>timmins, ontario, canada</td>\n",
       "      <td>34.7439</td>\n",
       "      <td>0002005018</td>\n",
       "      <td>5</td>\n",
       "      <td>Clara Callan</td>\n",
       "      <td>Richard Bruce Wright</td>\n",
       "      <td>2001.0</td>\n",
       "      <td>HarperFlamingo Canada</td>\n",
       "      <td>http://images.amazon.com/images/P/0002005018.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0002005018.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0002005018.0...</td>\n",
       "      <td>In a small town in Canada, Clara Callan reluct...</td>\n",
       "      <td>en</td>\n",
       "      <td>[Actresses]</td>\n",
       "      <td>timmins</td>\n",
       "      <td>ontario</td>\n",
       "      <td>canada</td>\n",
       "      <td>In a small town in Canada, Clara Callan reluct...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>8</td>\n",
       "      <td>timmins, ontario, canada</td>\n",
       "      <td>34.7439</td>\n",
       "      <td>0060973129</td>\n",
       "      <td>0</td>\n",
       "      <td>Decision in Normandy</td>\n",
       "      <td>Carlo D'Este</td>\n",
       "      <td>1991.0</td>\n",
       "      <td>HarperPerennial</td>\n",
       "      <td>http://images.amazon.com/images/P/0060973129.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0060973129.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0060973129.0...</td>\n",
       "      <td>Here, for the first time in paperback, is an o...</td>\n",
       "      <td>en</td>\n",
       "      <td>[1940-1949]</td>\n",
       "      <td>timmins</td>\n",
       "      <td>ontario</td>\n",
       "      <td>canada</td>\n",
       "      <td>Here, for the first time in paperback, is an o...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8</td>\n",
       "      <td>timmins, ontario, canada</td>\n",
       "      <td>34.7439</td>\n",
       "      <td>0374157065</td>\n",
       "      <td>0</td>\n",
       "      <td>Flu: The Story of the Great Influenza Pandemic...</td>\n",
       "      <td>Gina Bari Kolata</td>\n",
       "      <td>1999.0</td>\n",
       "      <td>Farrar Straus Giroux</td>\n",
       "      <td>http://images.amazon.com/images/P/0374157065.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0374157065.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0374157065.0...</td>\n",
       "      <td>Describes the great flu epidemic of 1918, an o...</td>\n",
       "      <td>en</td>\n",
       "      <td>[Medical]</td>\n",
       "      <td>timmins</td>\n",
       "      <td>ontario</td>\n",
       "      <td>canada</td>\n",
       "      <td>Describes the great flu epidemic of 1918, an o...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>8</td>\n",
       "      <td>timmins, ontario, canada</td>\n",
       "      <td>34.7439</td>\n",
       "      <td>0393045218</td>\n",
       "      <td>0</td>\n",
       "      <td>The Mummies of Urumchi</td>\n",
       "      <td>E. J. W. Barber</td>\n",
       "      <td>1999.0</td>\n",
       "      <td>W. W. Norton &amp; Company</td>\n",
       "      <td>http://images.amazon.com/images/P/0393045218.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0393045218.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0393045218.0...</td>\n",
       "      <td>A look at the incredibly well-preserved ancien...</td>\n",
       "      <td>en</td>\n",
       "      <td>[Design]</td>\n",
       "      <td>timmins</td>\n",
       "      <td>ontario</td>\n",
       "      <td>canada</td>\n",
       "      <td>A look at the incredibly well-preserved ancien...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   user_id                   location      age        isbn  rating  \\\n",
       "0        2  stockton, california, usa  18.0000  0195153448       0   \n",
       "1        8   timmins, ontario, canada  34.7439  0002005018       5   \n",
       "2        8   timmins, ontario, canada  34.7439  0060973129       0   \n",
       "3        8   timmins, ontario, canada  34.7439  0374157065       0   \n",
       "4        8   timmins, ontario, canada  34.7439  0393045218       0   \n",
       "\n",
       "                                          book_title           book_author  \\\n",
       "0                                Classical Mythology    Mark P. O. Morford   \n",
       "1                                       Clara Callan  Richard Bruce Wright   \n",
       "2                               Decision in Normandy          Carlo D'Este   \n",
       "3  Flu: The Story of the Great Influenza Pandemic...      Gina Bari Kolata   \n",
       "4                             The Mummies of Urumchi       E. J. W. Barber   \n",
       "\n",
       "   year_of_publication                publisher  \\\n",
       "0               2002.0  Oxford University Press   \n",
       "1               2001.0    HarperFlamingo Canada   \n",
       "2               1991.0          HarperPerennial   \n",
       "3               1999.0     Farrar Straus Giroux   \n",
       "4               1999.0   W. W. Norton & Company   \n",
       "\n",
       "                                               img_s  \\\n",
       "0  http://images.amazon.com/images/P/0195153448.0...   \n",
       "1  http://images.amazon.com/images/P/0002005018.0...   \n",
       "2  http://images.amazon.com/images/P/0060973129.0...   \n",
       "3  http://images.amazon.com/images/P/0374157065.0...   \n",
       "4  http://images.amazon.com/images/P/0393045218.0...   \n",
       "\n",
       "                                               img_m  \\\n",
       "0  http://images.amazon.com/images/P/0195153448.0...   \n",
       "1  http://images.amazon.com/images/P/0002005018.0...   \n",
       "2  http://images.amazon.com/images/P/0060973129.0...   \n",
       "3  http://images.amazon.com/images/P/0374157065.0...   \n",
       "4  http://images.amazon.com/images/P/0393045218.0...   \n",
       "\n",
       "                                               img_l  \\\n",
       "0  http://images.amazon.com/images/P/0195153448.0...   \n",
       "1  http://images.amazon.com/images/P/0002005018.0...   \n",
       "2  http://images.amazon.com/images/P/0060973129.0...   \n",
       "3  http://images.amazon.com/images/P/0374157065.0...   \n",
       "4  http://images.amazon.com/images/P/0393045218.0...   \n",
       "\n",
       "                                             Summary Language  \\\n",
       "0  Provides an introduction to classical myths pl...       en   \n",
       "1  In a small town in Canada, Clara Callan reluct...       en   \n",
       "2  Here, for the first time in paperback, is an o...       en   \n",
       "3  Describes the great flu epidemic of 1918, an o...       en   \n",
       "4  A look at the incredibly well-preserved ancien...       en   \n",
       "\n",
       "           Category      city       state country  \\\n",
       "0  [Social Science]  stockton  california     usa   \n",
       "1       [Actresses]   timmins     ontario  canada   \n",
       "2       [1940-1949]   timmins     ontario  canada   \n",
       "3         [Medical]   timmins     ontario  canada   \n",
       "4          [Design]   timmins     ontario  canada   \n",
       "\n",
       "                                            combined  \n",
       "0  Provides an introduction to classical myths pl...  \n",
       "1  In a small town in Canada, Clara Callan reluct...  \n",
       "2  Here, for the first time in paperback, is an o...  \n",
       "3  Describes the great flu epidemic of 1918, an o...  \n",
       "4  A look at the incredibly well-preserved ancien...  "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Jumlah data setelah menghapus duplikasi: 136256\n"
     ]
    }
   ],
   "source": [
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.neighbors import NearestNeighbors\n",
    "\n",
    "# ✅ 0. Filter data yang tidak valid SEBELUM gabungkan teks\n",
    "\n",
    "# Hapus entri dengan ringkasan tidak valid (misal: hanya berisi '9')\n",
    "preprocessed_clean = preprocessed_clean[preprocessed_clean['Summary'].str.strip() != '9']\n",
    "\n",
    "# Hapus entri dengan kategori kosong (misal: [])\n",
    "preprocessed_clean = preprocessed_clean[preprocessed_clean['Category'].astype(str) != '[]']\n",
    "\n",
    "# Hapus entri dengan ringkasan terlalu pendek (< 30 karakter)\n",
    "preprocessed_clean = preprocessed_clean[preprocessed_clean['Summary'].str.len() > 30]\n",
    "\n",
    "# Hapus duplikasi berdasarkan book_title + publisher + Summary\n",
    "preprocessed_clean = preprocessed_clean.drop_duplicates(subset=['book_title', 'publisher', 'Summary'])\n",
    "\n",
    "# ✅ 1. Gabungkan beberapa kolom jadi satu fitur teks (Summary + Category + book_title + Publisher)\n",
    "preprocessed_clean['combined'] = (\n",
    "    preprocessed_clean['Summary'].fillna('') + ' ' +\n",
    "    preprocessed_clean['Category'].astype(str) + ' ' +\n",
    "    preprocessed_clean['book_title'].fillna('') + ' ' +\n",
    "    preprocessed_clean['publisher'].fillna('')\n",
    ")\n",
    "\n",
    "# 2. Tampilkan data awal (opsional)\n",
    "display(preprocessed_clean.head())\n",
    "\n",
    "# ✅ 3. Hapus duplikasi berdasarkan kolom gabungan (jika masih ada)\n",
    "preprocessed_clean = preprocessed_clean.drop_duplicates(subset='combined')\n",
    "\n",
    "# ✅ 3.1 Reset index agar tfidf_matrix sesuai dengan DataFrame\n",
    "preprocessed_clean = preprocessed_clean.reset_index(drop=True)\n",
    "\n",
    "# 4. Cek jumlah data setelah menghapus duplikasi\n",
    "print(f\"Jumlah data setelah menghapus duplikasi: {len(preprocessed_clean)}\")\n",
    "\n",
    "# 5. Inisialisasi TF-IDF Vectorizer\n",
    "tfidf = TfidfVectorizer(stop_words='english')\n",
    "\n",
    "# 6. Transformasi ke matriks TF-IDF\n",
    "tfidf_matrix = tfidf.fit_transform(preprocessed_clean['combined'])\n",
    "\n",
    "# 7. Buat model Nearest Neighbors berbasis cosine similarity\n",
    "nn_model = NearestNeighbors(metric='cosine', algorithm='brute')\n",
    "nn_model.fit(tfidf_matrix)\n",
    "\n",
    "# 8. Buat mapping judul → index\n",
    "indices = pd.Series(preprocessed_clean.index, index=preprocessed_clean['book_title'].str.lower()).drop_duplicates()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2937cf1d-5802-405a-9489-c7dc439b2050",
   "metadata": {},
   "source": [
    "# ✅ Kesimpulan Tahapan Content-Based Filtering\n",
    "\n",
    " 1. Pembersihan dan Filter Data\n",
    " - Menghapus entri dengan ringkasan tidak valid ('9')\n",
    " - Menghapus entri dengan kategori kosong ([])\n",
    " - Menghapus ringkasan yang terlalu pendek (< 30 karakter)\n",
    " - Menghapus duplikasi berdasarkan book_title + publisher + Summary\n",
    "\n",
    " 2. Penggabungan Fitur Teks\n",
    " - Menggabungkan kolom Summary, Category, book_title, dan publisher ke kolom 'combined'\n",
    " - Ini bertujuan menyatukan informasi penting dalam satu representasi teks yang kaya konten\n",
    "\n",
    " 3. Penghapusan Duplikasi dan Reset Index\n",
    " - Membersihkan data dari duplikasi berdasarkan kolom 'combined'\n",
    " - Reset index untuk memastikan baris DataFrame sejajar dengan tfidf_matrix\n",
    "\n",
    " 4. TF-IDF Vectorization\n",
    " - Mengubah data teks pada kolom 'combined' menjadi vektor numerik dengan TfidfVectorizer\n",
    " - TF-IDF membantu menekankan kata-kata penting dan menurunkan bobot kata umum\n",
    "\n",
    " 5. Pelatihan Model Nearest Neighbors\n",
    " - Melatih model NearestNeighbors dengan tfidf_matrix\n",
    " - Menggunakan cosine similarity untuk mengukur kemiripan antar buku\n",
    "\n",
    " 6. Pemetaan Judul Buku ke Index\n",
    " - Membuat mapping dari judul buku (dalam lowercase) ke index DataFrame\n",
    " - Memungkinkan akses langsung ke representasi vektor berdasarkan input judul buku\n",
    "\n",
    " 📌 Hasil Akhir\n",
    " Sistem rekomendasi content-based siap digunakan untuk menyarankan buku yang mirip\n",
    " berdasarkan isi dan metadata, tanpa bergantung pada rating pengguna lain.\n",
    " Sangat cocok untuk cold-start scenario seperti buku baru yang belum memiliki rating.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "66239aca-93d8-40ac-91eb-fd6c16841def",
   "metadata": {},
   "source": [
    "## B. Model and Result"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c1e8c4d6-9c73-40a5-afdd-eca4f0aea2d4",
   "metadata": {},
   "source": [
    "Menemukan buku-buku yang paling mirip dengan buku input, sehingga bisa memberikan rekomendasi yang relevan secara konten, meskipun belum pernah diberi rating oleh pengguna."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "id": "853e7f61-615b-4c2c-8320-9eb3c564f49d",
   "metadata": {},
   "outputs": [],
   "source": [
    "def recommend_books_nn(title, df=preprocessed_clean, top_n=5):\n",
    "    title = title.lower()\n",
    "\n",
    "    if title not in indices:\n",
    "        return f\"Buku dengan judul '{title}' tidak ditemukan.\"\n",
    "\n",
    "    idx = indices[title]\n",
    "\n",
    "    # Vektorisasi buku input\n",
    "    book_vector = tfidf_matrix[idx]\n",
    "\n",
    "    # Temukan top-N buku mirip\n",
    "    distances, indices_nn = nn_model.kneighbors(book_vector, n_neighbors=top_n+1)\n",
    "\n",
    "    # Ambil hasil rekomendasi (kecuali dirinya sendiri)\n",
    "    rec_indices = indices_nn[0][1:]\n",
    "\n",
    "    return df[['book_title', 'Category', 'Summary', 'publisher']].iloc[rec_indices]\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e6bf2e99-dd47-48b5-9a12-ab4216b39e8a",
   "metadata": {},
   "source": [
    "\n",
    "Penjelasan Fungsi:\n",
    "1. Fungsi menerima judul buku yang ingin dicari kemiripannya.\n",
    "2. Menggunakan TF-IDF untuk mengukur kemiripan konten antar buku.\n",
    "3. Model NearestNeighbors digunakan untuk mencari buku dengan vektor TF-IDF paling dekat (mirip).\n",
    "4. Mengembalikan beberapa buku yang memiliki konten mirip berdasarkan ringkasan, kategori, judul, dan penerbit.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a11c665f-8b68-4b20-a4d3-910e4d53cf4f",
   "metadata": {},
   "source": [
    "## C. Testing System Recommendation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "id": "ad817ebc-7c31-4f32-9089-bcfc599f19a7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>book_title</th>\n",
       "      <th>Category</th>\n",
       "      <th>Summary</th>\n",
       "      <th>publisher</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>7107</th>\n",
       "      <td>The Secret Life of Bees</td>\n",
       "      <td>[Fiction]</td>\n",
       "      <td>After her &amp;quot;stand-in mother,&amp;quot; a bold ...</td>\n",
       "      <td>Viking Books</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>66528</th>\n",
       "      <td>Black Boy</td>\n",
       "      <td>[African American authors]</td>\n",
       "      <td>Relates what it was like for a Black child in ...</td>\n",
       "      <td>Harpercollins Publisher</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>119670</th>\n",
       "      <td>Midnight Heat</td>\n",
       "      <td>[Fiction]</td>\n",
       "      <td>Outraged by the poverty and injustice all arou...</td>\n",
       "      <td>Kensington Pub Corp (Mm)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21392</th>\n",
       "      <td>Clover</td>\n",
       "      <td>[Fiction]</td>\n",
       "      <td>After her father dies within hours of being ma...</td>\n",
       "      <td>Ballantine Books</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>85457</th>\n",
       "      <td>LIBRARY OF CLASSIC CHILDREN'S LITERATURE</td>\n",
       "      <td>[Juvenile Fiction]</td>\n",
       "      <td>Presents some of the classics of children&amp;#39;...</td>\n",
       "      <td>Courage Books</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                      book_title                    Category  \\\n",
       "7107                     The Secret Life of Bees                   [Fiction]   \n",
       "66528                                  Black Boy  [African American authors]   \n",
       "119670                             Midnight Heat                   [Fiction]   \n",
       "21392                                     Clover                   [Fiction]   \n",
       "85457   LIBRARY OF CLASSIC CHILDREN'S LITERATURE          [Juvenile Fiction]   \n",
       "\n",
       "                                                  Summary  \\\n",
       "7107    After her &quot;stand-in mother,&quot; a bold ...   \n",
       "66528   Relates what it was like for a Black child in ...   \n",
       "119670  Outraged by the poverty and injustice all arou...   \n",
       "21392   After her father dies within hours of being ma...   \n",
       "85457   Presents some of the classics of children&#39;...   \n",
       "\n",
       "                       publisher  \n",
       "7107                Viking Books  \n",
       "66528    Harpercollins Publisher  \n",
       "119670  Kensington Pub Corp (Mm)  \n",
       "21392           Ballantine Books  \n",
       "85457              Courage Books  "
      ]
     },
     "execution_count": 96,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "recommend_books_nn(\"The Secret Life of Bees\", top_n=5)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3450fbed-0bba-441f-b54e-08d8397c0050",
   "metadata": {},
   "source": [
    "### Penjelasan Output Rekomendasi: \"The Secret Life of Bees\"\n",
    "\n",
    "Hasil dari pemanggilan fungsi `recommend_books_nn(\"The Secret Life of Bees\", top_n=5)` memberikan 5 buku yang direkomendasikan berdasarkan kemiripan konten dengan buku input. Rekomendasi ini dihasilkan dari model content-based filtering yang membandingkan fitur teks gabungan (summary, kategori, judul, dan penerbit).\n",
    "\n",
    "Berikut adalah penjelasan untuk masing-masing buku hasil rekomendasi:\n",
    "\n",
    "1. **Black Boy**\n",
    "   - **Kategori**: [African American authors]\n",
    "   - **Alasan Rekomendasi**: Buku ini memiliki konten naratif yang kuat tentang pengalaman hidup, mirip dengan tema pencarian identitas dalam *The Secret Life of Bees*.\n",
    "\n",
    "2. **Midnight Heat**\n",
    "   - **Kategori**: [Fiction]\n",
    "   - **Alasan Rekomendasi**: Novel fiksi dengan elemen emosi kuat dan konflik sosial, cocok dengan pembaca yang menyukai dinamika karakter dan perjuangan batin.\n",
    "\n",
    "3. **Clover**\n",
    "   - **Kategori**: [Fiction]\n",
    "   - **Alasan Rekomendasi**: Sama-sama menceritakan hubungan keluarga dan trauma kehilangan, dengan karakter wanita muda sebagai tokoh utama.\n",
    "\n",
    "4. **LIBRARY OF CLASSIC CHILDREN'S LITERATURE**\n",
    "   - **Kategori**: [Juvenile Fiction]\n",
    "   - **Alasan Rekomendasi**: Meskipun ditujukan untuk anak-anak, rekomendasi muncul karena kemiripan gaya naratif dan nilai-nilai emosional yang terkandung dalam cerita.\n",
    "\n",
    "5. **The Secret Life of Bees**\n",
    "   - **Kategori**: [Fiction]\n",
    "   - **Catatan**: Buku ini sendiri tetap muncul sebagai hasil teratas, karena secara teknis memiliki kemiripan tertinggi dengan dirinya sendiri. Namun hasil akhir hanya menampilkan buku-buku selain buku input, sesuai logika dalam fungsi.\n",
    "\n",
    "### Kesimpulan\n",
    "Rekomendasi yang dihasilkan menunjukkan bahwa sistem dapat mengenali pola naratif, tema emosional, serta atribut tekstual lainnya yang relevan. Hal ini membuktikan bahwa pendekatan content-based cukup efektif meskipun tidak menggunakan data rating pengguna.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "id": "24e1edc8-5aea-4fdb-98a0-1ae86b55e356",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>book_title</th>\n",
       "      <th>Category</th>\n",
       "      <th>Summary</th>\n",
       "      <th>publisher</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>85070</th>\n",
       "      <td>A Painted House</td>\n",
       "      <td>[Fiction]</td>\n",
       "      <td>Racial tension, a forbidden love affair, and m...</td>\n",
       "      <td>Delta</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>133123</th>\n",
       "      <td>A Painted House (Limited Edition)</td>\n",
       "      <td>[Fiction]</td>\n",
       "      <td>Racial tension, a forbidden love affair, and m...</td>\n",
       "      <td>Doubleday</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>88762</th>\n",
       "      <td>Boy of the Painted Cave</td>\n",
       "      <td>[Juvenile Fiction]</td>\n",
       "      <td>Forbidden to make images, fourteen-year-old Ta...</td>\n",
       "      <td>Putnam Publishing Group</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>69752</th>\n",
       "      <td>River of Earth</td>\n",
       "      <td>[Fiction]</td>\n",
       "      <td>The chance of material prosperity lures a poor...</td>\n",
       "      <td>University Press of Kentucky</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>57278</th>\n",
       "      <td>The Wedding Dress</td>\n",
       "      <td>[Fiction]</td>\n",
       "      <td>Portraits of pioneer life are painted in eight...</td>\n",
       "      <td>Dell Publishing Company</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                               book_title            Category  \\\n",
       "85070                     A Painted House           [Fiction]   \n",
       "133123  A Painted House (Limited Edition)           [Fiction]   \n",
       "88762             Boy of the Painted Cave  [Juvenile Fiction]   \n",
       "69752                      River of Earth           [Fiction]   \n",
       "57278                   The Wedding Dress           [Fiction]   \n",
       "\n",
       "                                                  Summary  \\\n",
       "85070   Racial tension, a forbidden love affair, and m...   \n",
       "133123  Racial tension, a forbidden love affair, and m...   \n",
       "88762   Forbidden to make images, fourteen-year-old Ta...   \n",
       "69752   The chance of material prosperity lures a poor...   \n",
       "57278   Portraits of pioneer life are painted in eight...   \n",
       "\n",
       "                           publisher  \n",
       "85070                          Delta  \n",
       "133123                     Doubleday  \n",
       "88762        Putnam Publishing Group  \n",
       "69752   University Press of Kentucky  \n",
       "57278        Dell Publishing Company  "
      ]
     },
     "execution_count": 99,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "recommend_books_nn(\"A Painted House\", top_n=5)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0f3fff93-b7d2-48fd-b936-81e4cfa1c0ae",
   "metadata": {},
   "source": [
    "### 📚 Recommendation Output\n",
    "\n",
    "#### Judul: *A Painted House*\n",
    "\n",
    "Berdasarkan pendekatan content-based filtering, sistem memberikan rekomendasi buku yang memiliki kemiripan konten dengan *A Painted House* melalui analisis terhadap *summary*, *category*, *judul*, dan *publisher*. Berikut penjelasan hasil rekomendasinya:\n",
    "\n",
    "1. **A Painted House (Delta)** — Buku utama dan referensi pencarian. Kisahnya mengangkat tema ketegangan rasial dan cinta terlarang, dengan latar kehidupan petani.\n",
    "\n",
    "2. **A Painted House (Limited Edition) - Doubleday** — Edisi berbeda dari buku yang sama, tetap relevan karena memiliki isi konten dan cerita serupa.\n",
    "\n",
    "3. **Boy of the Painted Cave** — Memiliki elemen tematik serupa seperti pembatasan sosial dan perjuangan hidup di masyarakat tertutup, ditulis dalam genre *Juvenile Fiction*.\n",
    "\n",
    "4. **River of Earth** — Cerita tentang keluarga miskin yang berjuang untuk kehidupan lebih baik, serupa dengan latar ekonomi dan sosial di *A Painted House*.\n",
    "\n",
    "5. **The Wedding Dress** — Menceritakan potret kehidupan perintis dengan gaya naratif yang mirip, menggambarkan masa lalu dan perjuangan hidup di era tertentu.\n",
    "\n",
    "#### Kesimpulan:\n",
    "Rekomendasi ini menunjukkan bahwa sistem berhasil mengidentifikasi buku-buku dengan kemiripan tema, genre, dan latar suasana, bahkan meskipun judul atau pengarang berbeda. Ini menegaskan keefektifan pendekatan content-based dalam menemukan buku sejenis berdasarkan isi konten, bukan hanya popularitas atau rating pengguna lain.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7ea4ffd9-25d6-45d6-9594-a123b112b2c6",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3.10 (tfenv)",
   "language": "python",
   "name": "tfenv"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.18"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
