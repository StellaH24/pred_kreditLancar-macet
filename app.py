
# Ini adalah script Streamlit untuk aplikasi prediksi status kredit.

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Muat preprocessor dan model yang sudah disimpan
try:
    preprocessor = joblib.load('preprocessor.joblib')
    rf_model = joblib.load('rf_model.joblib')
    logreg_model = joblib.load('logreg_model.joblib')
except FileNotFoundError:
    st.error("File model atau preprocessor tidak ditemukan. Pastikan 'preprocessor.joblib', 'rf_model.joblib', dan 'logreg_model.joblib' berada di direktori yang sama dengan script ini.")
    st.stop() # Hentikan eksekusi jika file tidak ditemukan

st.title("Aplikasi Prediksi Status Kredit")

st.write("Masukkan data calon kreditur baru untuk mendapatkan prediksi status kredit (Lancar/Kredit Macet).")

# --- Bagian Input Data Kreditur Baru ---
# Menggunakan daftar kategori statis karena df dan X tidak tersedia di lingkungan Streamlit
# Ganti dengan daftar kategori unik yang sebenarnya dari data Anda saat preprocessing
# Ini adalah contoh, Anda harus menggantinya dengan daftar lengkap dan benar
unique_wilayah = ['SOFLB', 'SOKAN', 'SOAMK', 'SOUBT', 'SOUJT', 'SOBAT'] # Ganti dengan daftar lengkap
unique_pekerjaan = ['PEGAWAI SWASTA', 'IRT', 'PEDAGANG/WIRASWASTA', 'PETANI', 'PELAJAR/MAHASISWA', 'PNS', 'TNI/POLRI', 'LAINNYA'] # Ganti dengan daftar lengkap
unique_tempattinggal = ['MILIK SENDIRI', 'RUMAH SENDIRI', 'ORANG TUA', 'MILIK ORANG TUA', 'KONTRAK', 'LAINNYA'] # Ganti dengan daftar lengkap
unique_tipemotor = ['HY2', 'LF0', 'GF4', 'LD0', 'HZ1', 'GF3', 'LB2', 'LA4', 'LA5', 'LF5', 'LC6', 'LF8', 'LF4', 'LB1', 'LA6', 'LC5', 'LC1', 'LC2', 'LA1', 'HF1', 'HF3', 'LA2', 'HF2', 'HF4', 'LC3', 'LC4', 'LF1', 'LF2', 'LF3', 'LA3', 'LC7', 'LF6', 'LF7'] # Ganti dengan daftar lengkap
unique_warna = ['RD', 'BK', 'SV', 'WH', 'BL', 'MR', 'GR', 'OR', 'BR', 'YL', 'PURPLE', 'OTHERS'] # Ganti dengan daftar lengkap
unique_jkelamin = ['PRIA', 'WANITA'] # Berdasarkan info dataset
# KODEPOS, KECAMATAN, KELURAHAN akan dihandle sebagai string input teks

col1, col2 = st.columns(2)

with col1:
    wilayah = st.selectbox("WILAYAH", unique_wilayah)
    umur = st.number_input("UMUR", min_value=0, value=30)
    pekerjaan = st.selectbox("PEKERJAAN", unique_pekerjaan)
    tempattinggal = st.selectbox("TEMPATTINGGAL", unique_tempattinggal)
    penghasilan = st.number_input("PENGHASILAN", min_value=0, value=5000000)
    tipemotor = st.selectbox("TIPEMOTOR", unique_tipemotor)


with col2:
    warna = st.selectbox("WARNA", unique_warna)
    hargaotr = st.number_input("HARGAOTR", min_value=0, value=15000000)
    dp = st.number_input("DP", min_value=0, value=3000000)
    bunga = st.number_input("BUNGA", min_value=0.0, format="%.4f", value=20.0) # Gunakan format untuk float
    cicilan = st.number_input("CICILAN", min_value=0, value=1000000)
    kodepos = st.text_input("KODEPOS") # Treat as string for encoding
    kecamatan = st.text_input("KECAMATAN") # Treat as string for encoding
    kelurahan = st.text_input("KELURAHAN") # Treat as string for encoding
    jkelamin = st.selectbox("JKELAMIN", unique_jkelamin)


# Buat DataFrame dari input user
# Pastikan urutan kolom sesuai dengan urutan kolom di X sebelum preprocessing
# dan tipe datanya juga sesuai (string untuk kategorikal yang di-encode, number untuk numerik)
# Urutan kolom harus sama dengan X.columns saat preprocessor dilatih.
# Kita definisikan urutan kolom secara statis.
original_X_columns = ['WILAYAH', 'UMUR', 'PEKERJAAN', 'TEMPATTINGGAL', 'PENGHASILAN', 'TIPEMOTOR',
                      'WARNA', 'HARGAOTR', 'DP', 'BUNGA', 'CICILAN', 'KODEPOS', 'KECAMATAN',
                      'KELURAHAN', 'JKELAMIN'] # Urutan kolom yang benar sesuai preprocessing

user_data_dict = {
    'WILAYAH': [wilayah],
    'UMUR': [umur],
    'PEKERJAAN': [pekerjaan],
    'TEMPATTINGGAL': [tempattinggal],
    'PENGHASILAN': [penghasilan],
    'TIPEMOTOR': [tipemotor],
    'WARNA': [warna],
    'HARGAOTR': [hargaotr],
    'DP': [dp],
    'BUNGA': [bunga],
    'CICILAN': [cicilan],
    'KODEPOS': [kodepos],
    'KECAMATAN': [kecamatan],
    'KELURAHAN': [kelurahan],
    'JKELAMIN': [jkelamin]
}

# Pastikan kolom kategorikal yang awalnya numerik diubah ke string di DataFrame input
user_data = pd.DataFrame(user_data_dict)

for col in ['KODEPOS', 'KECAMATAN', 'KELURAHAN']:
     user_data[col] = user_data[col].astype(str)

# Pastikan urutan kolom user_data sama dengan urutan kolom saat preprocessor dilatih
user_data = user_data[original_X_columns]


# Tombol Prediksi
if st.button("Prediksi Status Kredit"):
    # Pra-pemrosesan data input user
    # Gunakan preprocessor yang sudah dilatih
    try:
        user_data_processed = preprocessor.transform(user_data)

        # Lakukan prediksi menggunakan model Random Forest
        prediction_rf = rf_model.predict(user_data_processed)
        prediction_status = "KREDIT MACET" if prediction_rf[0] == 1 else "LANCAR"

        # Tampilkan tabel informasi dan prediksi
        st.subheader("Hasil Prediksi")
        result_df = user_data.copy() # Gunakan user_data sebelum preprocessing
        result_df['PREDIKSI_STATUS'] = prediction_status
        st.table(result_df)

        # Simpan data input user yang sudah diproses untuk tombol probabilitas di session state
        # Session state diperlukan di Streamlit untuk menyimpan data antar interaksi
        st.session_state['user_data_for_prob'] = user_data_processed

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses input atau melakukan prediksi: {e}")
        st.write("Mohon periksa kembali input Anda.")


# Tombol untuk melihat persentase kredit macet (probabilitas)
# Tombol ini hanya muncul jika prediksi status sudah dilakukan
if 'user_data_for_prob' in st.session_state:
    if st.button("Lihat Persentase Kredit Macet"):
        try:
            # Gunakan model Regresi Logistik untuk mendapatkan probabilitas
            probability = logreg_model.predict_proba(st.session_state['user_data_for_prob'])[:, 1]
            probability_percentage = probability[0] * 100

            st.subheader("Probabilitas Kredit Macet")
            st.write(f"Probabilitas kredit macet untuk kreditur ini adalah: **{probability_percentage:.2f}%**")

        except Exception as e:
             st.error(f"Terjadi kesalahan saat menghitung probabilitas: {e}")

# --- Instruksi Deployment (Optional di script akhir) ---
# st.sidebar.header("Instruksi Deployment")
# st.sidebar.info(...)
