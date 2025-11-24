################################### 
# Ini Buat test gambar dari internet 
###################################
 

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

# ==========================================
# 1. KONFIGURASI
# ==========================================
# Ganti dengan nama file gambar yang Anda download
nama_file_gambar = 'upload gambar\sehat.webp' 

# Path ke model yang sudah ditraining
path_model = 'model_daun_apel.h5'

# Ukuran gambar WAJIB SAMA dengan saat training (128x128)
IMG_SIZE = (128, 128)

# Daftar Nama Kelas (Harus urut abjad sesuai folder training tadi)
# 0: Scab, 1: Black Rot, 2: Rust, 3: Healthy
class_names = ['Apple Scab', 'Black Rot', 'Cedar Apple Rust', 'Healthy']

# ==========================================
# 2. PREPROCESSING GAMBAR
# ==========================================
print(f"--- Memproses gambar: {nama_file_gambar} ---")

try:
    # Load gambar dan paksa ubah ukuran ke 128x128
    img = load_img(nama_file_gambar, target_size=IMG_SIZE)
    
    # Ubah menjadi array angka
    img_array = img_to_array(img)
    
    # PENTING: Tambahkan dimensi batch
    # Model butuh input (1, 128, 128, 3), bukan cuma (128, 128, 3)
    img_array = tf.expand_dims(img_array, 0) 

    # ==========================================
    # 3. PREDIKSI
    # ==========================================
    # Load Model
    model = tf.keras.models.load_model(path_model)
    
    # Lakukan prediksi
    predictions = model.predict(img_array)
    
    # Ambil hasil probabilitas tertinggi
    score = tf.nn.softmax(predictions[0])
    class_index = np.argmax(predictions[0])
    
    label_prediksi = class_names[class_index]
    confidence = 100 * np.max(predictions[0])

    print(f"\n✅ Hasil Prediksi: {label_prediksi}")
    print(f"📊 Keyakinan (Confidence): {confidence:.2f}%")

    # ==========================================
    # 4. TAMPILKAN HASIL
    # ==========================================
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Prediksi: {label_prediksi}\nYakin: {confidence:.2f}%", 
              color='green' if confidence > 70 else 'red')
    plt.axis("off")
    plt.show()

except FileNotFoundError:
    print(f"❌ ERROR: File '{nama_file_gambar}' tidak ditemukan di folder ini!")
    print("Pastikan Anda sudah menaruh gambar dan namanya benar.")
except Exception as e:
    print(f"Terjadi kesalahan: {e}")