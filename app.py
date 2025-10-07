# --- app.py ---

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =====================================
# 1️⃣ Load Model dan Preprocessor
# =====================================
try:
    preprocessor = joblib.load('preprocessor.joblib')
    rf_model = joblib.load('rf_model.joblib')
    logreg_model = joblib.load('logreg_model.joblib')
except FileNotFoundError:
    st.error("❌ File model atau preprocessor tidak ditemukan. Pastikan file .joblib ada di direktori yang sama dengan app.py.")
    st.stop()

st.title("📊 Aplikasi Prediksi Status Kredit")
st.write("Masukkan data calon kreditur baru untuk memprediksi status kredit (Lancar / Kredit Macet).")

# =====================================
# 2️⃣ Input Data dari Pengguna
# =====================================

# Daftar kategori statis (karena df tidak tersedia di Streamlit)
unique_wilayah = ['SOFLB', 'SOKAN', 'SOAMK', 'SOUBT', 'SOUJT', 'SOBAT']
unique_pekerjaan = ['PEGAWAI SWASTA', 'IRT', 'PEDAGANG/WIRASWASTA', 'PETANI', 'PNS', 'TNI/POLRI', 'LAINNYA']
unique_tempattinggal = ['MILIK SENDIRI', 'RUMAH SENDIRI', 'ORANG TUA', 'MILIK ORANG TUA', 'KONTRAK', 'LAINNYA']
unique_tipemotor = ['HY2', 'LF0', 'GF4', 'LD0', 'HZ1', 'GF3', 'LB2', 'LA4', 'LF1', 'LF2']
unique_warna = ['RD', 'BK', 'SV', 'WH', 'BL', 'MR', 'GR', 'OR', 'YL', 'LAINNYA']
unique_jkelamin = ['PRIA', 'WANITA']

col1, col2 = st.columns(2)

with col1:
    wilayah = st.selectbox("WILAYAH", unique_wilayah)
    umur = st.number_input("UMUR", min_value=18, max_value=70, value=30)
    pekerjaan = st.selectbox("PEKERJAAN", unique_pekerjaan)
    tempattinggal = st.selectbox("TEMPATTINGGAL", unique_tempattinggal)
    penghasilan = st.number_input("PENGHASILAN (Rp)", min_value=0, value=5000000)
    tipemotor = st.selectbox("TIPEMOTOR", unique_tipemotor)

with col2:
    warna = st.selectbox("WARNA", unique_warna)
    hargaotr = st.number_input("HARGAOTR (Rp)", min_value=0, value=20000000)
    dp = st.number_input("DP (Rp)", min_value=0, value=5000000)
    bunga = st.number_input("BUNGA (%)", min_value=0.0, value=20.0, format="%.2f")
    cicilan = st.number_input("CICILAN (Rp)", min_value=0, value=1000000)
    kodepos = st.text_input("KODEPOS")
    kecamatan = st.text_input("KECAMATAN")
    kelurahan = st.text_input("KELURAHAN")
    jkelamin = st.selectbox("JENIS KELAMIN", unique_jkelamin)

# Buat DataFrame dari input
user_data = pd.DataFrame({
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
    'KODEPOS': [str(kodepos)],
    'KECAMATAN': [str(kecamatan)],
    'KELURAHAN': [str(kelurahan)],
    'JKELAMIN': [jkelamin]
})

# =====================================
# 3️⃣ Tombol Prediksi
# =====================================
if st.button("🔍 Prediksi Status Kredit"):
    try:
        # Gunakan preprocessor yang sudah disimpan
        user_data_processed = preprocessor.transform(user_data)

        # Prediksi dengan Random Forest
        prediction_rf = rf_model.predict(user_data_processed)[0]
        pred_label = "KREDIT MACET" if prediction_rf == 1 else "LANCAR"

        st.success(f"**Hasil Prediksi:** {pred_label}")

        # Simpan untuk probabilitas
        st.session_state['user_data_processed'] = user_data_processed

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

# =====================================
# 4️⃣ Tombol Probabilitas Kredit Macet
# =====================================
if 'user_data_processed' in st.session_state:
    if st.button("📈 Tampilkan Persentase Kredit Macet"):
        try:
            proba = logreg_model.predict_proba(st.session_state['user_data_processed'])[0][1]
            st.info(f"Probabilitas kredit macet: **{proba * 100:.2f}%**")
        except Exception as e:
            st.error(f"Gagal menghitung probabilitas: {e}")

st.caption("Aplikasi ini menggunakan model yang dilatih dengan Random Forest dan Logistic Regression.")
