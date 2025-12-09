# Tugas Besar IF5100 – Global Superstore

Repo ini dipakai untuk Tugas Besar IF5100 dengan dataset **Global Superstore**.

Peran tim:
- 👤 **Orang 1 – Data Analyst** (repo & kode ini)
- 👤 Orang 2 – Machine Learning Engineer
- 👤 Orang 3 – Project Manager & Reporter

---

## 1. Struktur Singkat

```text
data/
  raw/        -> Global_Superstore2.csv (data mentah)
  processed/  -> 
    global_superstore_clean.csv (data bersih)
    global_superstore_model_ready.csv (data siap modelling)
notebooks/
  01_data_prep_eda_viz.ipynb  -> kerjaan Orang 1
  02_modeling.ipynb           -> untuk modelling (Orang 2)
src/
  utils/io_utils.py
  data_prep/clean_global_superstore.py
  eda/eda_utils.py
reports/figures/              -> gambar-gambar visualisasi
```

---

## 2. Yang Sudah Dikerjakan Orang 1

Di `01_data_prep_eda_viz.ipynb` dan `src/data_prep/clean_global_superstore.py`:

* **Data Prep**

  * Load data mentah dari `data/raw/Global_Superstore2.csv`
  * Bersihkan:

    * standar nama kolom
    * parse tanggal (`order_date`, `ship_date`)
    * hapus duplikat
    * handle missing (baris dengan NA di `sales`, `quantity`, `profit`, `discount` di-drop)
    * drop `postal_code` (banyak NA, lokasi sudah terwakili kolom lain)
  * Tambah fitur:

    * `order_year`, `order_month`, `order_quarter`
    * `shipping_days` (selisih ship–order)
    * `profit_margin`, `sales_per_quantity`
  * Buat **target**:

    * `is_profitable = 1` jika `profit > 0`, else 0
  * Simpan ke: `data/processed/global_superstore_clean.csv`

* **EDA & Visualisasi**

  * `df.shape`, `df.info()`, `df.describe()`, nilai unik, dan missing
  * Plot:

    * Total sales per **category**
    * Sales & profit Top 10 **sub_category**
    * **Monthly sales trend**
    * **Discount vs profit** (scatter + trend line)
    * 1 plot interaktif Plotly: monthly sales trend
  * Gambar disimpan di `reports/figures/`

* **Dataset untuk Modelling**

  * Fungsi `encode_categoricals_for_model(df_clean)` menyiapkan data untuk prediksi `is_profitable`:

    * Drop ID & kolom bocor (`profit`, `profit_margin`) dan kolom high-cardinality (city, state, country, market, sub_category, tanggal mentah).
    * Fitur numerik: `sales`, `quantity`, `discount`, `shipping_cost`, `order_year`, `order_month`, `order_quarter`, `shipping_days`, `sales_per_quantity`
    * Fitur kategorikal:

      * `order_priority` → ordinal (Low<Medium<High<Critical)
      * `ship_mode`, `segment`, `region`, `category` → one-hot
    * Hasil akhir: semua fitur numerik + `is_profitable` sebagai target.

---

## 3. Petunjuk untuk Orang 2 (Modelling)

Buka `notebooks/02_modeling.ipynb`, di bagian paling atas sudah ada cell:

* Load `df_clean` (atau generate kalau belum ada)
* Panggil `encode_categoricals_for_model(df_clean)`
* Siapkan:

  * `X` = fitur
  * `y` = `is_profitable`

Setelah itu bisa lanjut:

* `train_test_split`, normalisasi/standardisasi (jika perlu)
* Latih model (misal: Logistic Regression, Random Forest, ensemble, dll.)
* Hitung metrik (Accuracy, Precision, Recall, F1, dll.)

---

## 4. Petunjuk untuk Orang 3 (Laporan & Video)

Dari kerjaan Orang 1, yang bisa dipakai:

* Deskripsi dataset (jumlah baris/kolom, sumber data)
* Langkah Data Preparation (ringkas seperti di atas)
* Penjelasan target `is_profitable`
* Penjelasan singkat fitur yang dipakai modelling
* Gambar-gambar di `reports/figures/` untuk dimasukkan ke laporan & slide
