# Tugas Besar IF5100 – Global Superstore

Repo ini dipakai untuk Tugas Besar IF5100 dengan dataset **Global Superstore**.

Peran tim:

* 👤 **Orang 1 – Data Analyst** → struktur repo, data preparation, EDA, visualisasi
* 👤 **Orang 2 – Machine Learning Engineer** → modelling, evaluasi model, interpretasi hasil
* 👤 **Orang 3 – Project Manager & Reporter** → koordinasi, laporan, slide, dan video

---

## 1. Struktur Singkat Repo

```text
data/
  raw/        -> Global_Superstore2.csv (data mentah)
  processed/  -> 
    global_superstore_clean.csv        (data bersih)
    global_superstore_model_ready.csv  (data siap modelling, sudah di-encode)

notebooks/
  01_data_prep_eda_viz.ipynb  -> kerjaan Orang 1 (sudah terisi untuk data prep & EDA)
  02_modeling.ipynb           -> untuk modelling (diisi Orang 2)

src/
  utils/
    io_utils.py                    -> helper untuk load/save data
  data_prep/
    clean_global_superstore.py     -> script cleaning + feature engineering
  eda/
    eda_utils.py                   -> fungsi EDA & plotting (dipakai di notebook)

reports/
  figures/                         -> gambar-gambar visualisasi (untuk laporan & slide)
````

Orang 2 dan Orang 3 cukup fokus ke `data/processed/`, `notebooks/`, dan `reports/`.

---

## 2. Yang Sudah Dikerjakan Orang 1 (Data Analyst)

Semua logic data prep dan EDA utama ada di:

* `notebooks/01_data_prep_eda_viz.ipynb`
* `src/data_prep/clean_global_superstore.py`
* `src/eda/eda_utils.py`

### 2.1 Data Preparation

Langkah yang sudah dilakukan:

* Load data mentah dari `data/raw/Global_Superstore2.csv`

* Cleaning:

  * Standarisasi nama kolom
  * Parse tanggal (`order_date`, `ship_date`) menjadi tipe datetime
  * Hapus baris duplikat
  * Tangani missing values:

    * Baris dengan NA di `sales`, `quantity`, `profit`, `discount` di-drop
  * Drop kolom `postal_code` (banyak NA dan lokasi sudah tercakup di kolom lain)

* Tambah fitur (feature engineering):

  * `order_year`, `order_month`, `order_quarter`
  * `shipping_days` (selisih hari antara `ship_date` dan `order_date`)
  * `profit_margin = profit / sales`
  * `sales_per_quantity = sales / quantity`

* Buat **target klasifikasi**:

  * `is_profitable = 1` jika `profit > 0`, else `0`

* Output:

  * Hasil cleaning disimpan di:
    `data/processed/global_superstore_clean.csv`

### 2.2 EDA & Visualisasi

Explorasi yang sudah dilakukan di `01_data_prep_eda_viz.ipynb`:

* Info awal dataset:

  * `df.shape`, `df.info()`, `df.describe()`
  * Cek nilai unik dan missing values per kolom

* Plot utama yang sudah dihasilkan:

  * Total `sales` per **category**
  * `sales` dan `profit` untuk Top 10 **sub_category**
  * **Monthly sales trend** (agregasi per bulan)
  * **Discount vs profit** (scatter plot + trend line sederhana)
  * Satu plot interaktif Plotly: monthly sales trend

* Semua gambar statis disimpan ke folder:

  * `reports/figures/`

Orang 3 bisa langsung pakai gambar ini untuk laporan dan slide, tanpa perlu run ulang.

### 2.3 Dataset Siap Modelling

Fungsi `encode_categoricals_for_model(df_clean)` di `src/data_prep/clean_global_superstore.py` menghasilkan dataset final untuk modelling:

* Output final untuk modelling:

  * `data/processed/global_superstore_model_ready.csv`
    → **file ini yang dipakai Orang 2 untuk seluruh eksperimen model**

Ringkasan fitur di `global_superstore_model_ready.csv`:

* **Target:**

  * `is_profitable` (1 = order untung, 0 = tidak untung)

* **Fitur numerik:**

  * `sales`, `quantity`, `discount`, `shipping_cost`
  * `order_year`, `order_month`, `order_quarter`
  * `shipping_days`, `sales_per_quantity`
  * `profit_margin` (boleh dipakai atau dibuang sesuai kebutuhan model)

* **Fitur kategorikal yang sudah di-encode:**

  * `order_priority` → ordinal encoding (Low < Medium < High < Critical)
  * `ship_mode`, `segment`, `region`, `category` → one-hot encoding
  * Kolom one-hot biasanya bernama `ship_mode_*`, `segment_*`, dst.

---

## 3. Petunjuk untuk Orang 2 – Machine Learning Engineer

File utama untuk modelling:

* `notebooks/02_modeling.ipynb`

### 3.1 Load Dataset Modelling

Contoh kode awal yang bisa langsung dipakai:

```python
import pandas as pd

df_model = pd.read_csv("data/processed/global_superstore_model_ready.csv")
TARGET_COL = "is_profitable"

X = df_model.drop(columns=[TARGET_COL])
y = df_model[TARGET_COL]
```

Split train–test:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### 3.2 Saran Alur Kerja Model

Minimal lakukan:

1. Cek distribusi label `is_profitable` (imbalance atau tidak)

2. Latih beberapa model baseline:

   * Logistic Regression
   * Decision Tree Classifier
   * Random Forest Classifier
   * Ensemble Models

3. Evaluasi dengan metrik:

   * Accuracy
   * Precision, Recall, F1-score
   * Confusion matrix

4. Simpan hasil penting:

   * Tabel ringkasan metrik per model
   * Confusion matrix (bisa simpan gambar ke `reports/figures/`)
   * Kalau sempat, feature importance (dari Tree/Random Forest)

### 3.3 Output yang Diharapkan dari Orang 2

* Notebook `02_modeling.ipynb` yang rapi dengan:

  * Penjelasan singkat tiap langkah
  * Kode training dan evaluasi beberapa model
  * Tabel ringkasan metrik

* Gambar:

  * Confusion matrix dan plot lain (jika ada) disimpan ke
    `reports/figures/` dengan nama yang jelas
    misal: `cm_logreg.png`, `cm_rf.png`, dll.

Orang 3 akan pakai metrik dan gambar ini untuk bagian hasil modelling di laporan dan slide.

---

## 4. Petunjuk untuk Orang 3 – Project Manager & Reporter

Fokus utama Orang 3:

* Menggabungkan hasil kerja Orang 1 dan Orang 2 menjadi:

  * Laporan tertulis
  * Slide presentasi
  * Script atau poin penting untuk video

### 4.1 Bahan dari Orang 1 (Data Prep & EDA)

Dari `01_data_prep_eda_viz.ipynb` dan `reports/figures/`:

* Deskripsi dataset:

  * Sumber (Global Superstore)
  * Jumlah baris dan kolom
  * Contoh kolom penting

* Langkah Data Preparation:

  * Cleaning apa saja yang dilakukan
  * Alasan membuang `postal_code`
  * Penjelasan pembuatan fitur baru
  * Penjelasan kenapa `is_profitable` dipakai sebagai target

* Visualisasi utama:

  * Perbandingan sales dan profit antar kategori
  * Tren penjualan bulanan
  * Hubungan discount dan profit

Semua gambar sudah ada di `reports/figures/`.

### 4.2 Bahan dari Orang 2 (Modelling)

Dari `02_modeling.ipynb`:

* Model apa saja yang dicoba
* Singkat tentang cara kerja model (garis besar saja)
* Tabel perbandingan metrik (accuracy, precision, recall, F1)
* Confusion matrix dan interpretasinya
* Model mana yang akhirnya dipilih dan kenapa

### 4.3 Hal yang Perlu Ditekankan di Laporan dan Video

* **Cerita alur proyek**:

  1. Masalah: ingin tahu faktor apa yang membuat sebuah order menguntungkan
  2. Data: Global Superstore, penjelasan singkat
  3. Data Preparation dan EDA (kerjaan Orang 1)
  4. Modelling dan hasilnya (kerjaan Orang 2)
  5. Kesimpulan dan insight bisnis

* **Hubungkan angka dengan bahasa awam**:

  * Contoh: “Model terbaik punya akurasi X persen, artinya dari 100 order, sekitar X order diprediksi dengan benar apakah menguntungkan atau tidak.”

* **Pastikan semua gambar di laporan dan slide** diberi:

  * Judul yang jelas
  * Caption singkat
  * Sumber: “diolah dari Global Superstore”

---

## 5. Ringkasannya

* Orang 1: sudah menyiapkan **data bersih + dataset siap modelling + EDA + visualisasi**.
* Orang 2: tinggal fokus ke **`02_modeling.ipynb`** untuk melatih dan evaluasi model menggunakan `global_superstore_model_ready.csv`.
* Orang 3: menyusun **laporan, slide, dan script video** dengan memanfaatkan:

  * Gambar di `reports/figures/`
  * Ringkasan langkah data prep dan EDA
  * Hasil dan interpretasi model dari Orang 2.
