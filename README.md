# Peramalan harga Bitcoin dengan PSO–ELM

Proyek skripsi Bagus Putra Sulung. Aplikasi Streamlit mereproduksi eksperimen Extreme Learning Machine (ELM) dan PSO–ELM dari studi yang diterbitkan di IDEALIS pada 2026, dengan penyimpanan eksperimen dan riwayat di SQLite. Data pasar diperoleh melalui yfinance ketika pengguna memilih memuat data.

- `research_engine` memuat rekayasa fitur, ELM, optimasi PSO, dan validasi walk-forward.
- `ui` memuat halaman data, eksperimen, hasil, dan prediksi.
- `services` dan `database` memuat penyimpanan data serta hasil eksperimen.
- `tests` memuat pengujian mesin peramalan dan penyimpanan aplikasi.

## Menjalankan aplikasi lokal

Gunakan Python 3.11. Dari PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .streamlit/secrets.toml.example .streamlit/secrets.toml
.\.venv\Scripts\python scripts/generate_password_hash.py
```

Ganti nilai `password_hash` pada `.streamlit/secrets.toml` dengan keluaran perintah terakhir. Jalankan `python -m streamlit run app.py --server.address 127.0.0.1` menggunakan Python dari lingkungan virtual dan masuk dengan akun yang Anda atur. SQLite dibuat saat aplikasi dijalankan. Untuk memulai dari data baru, gunakan menu data di aplikasi.

## Hasil penelitian dan batas evaluasi

Artikel melaporkan MAPE 2,15% pada skenario terbaik. Kode reproduksi menerapkan min–max sebelum pembagian walk-forward agar hasil sebanding dengan implementasi penelitian. Untuk mengukur kinerja pada data baru, ulangi evaluasi dengan skala yang dipelajari di setiap bagian latih. Basis data, model terlatih, dan konfigurasi akses dari arsip lokal tidak diperlukan untuk membuat basis data baru.

Penulis publikasi: Bagus Putra Sulung, Wiwit Agus Triyanto, dan Pratomo Setiaji. [Artikel dan DOI](https://doi.org/10.36080/idealis.v9i2.3812)
