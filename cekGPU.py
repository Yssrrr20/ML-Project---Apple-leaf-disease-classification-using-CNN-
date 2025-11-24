import tensorflow as tf

print("Versi TensorFlow:", tf.__version__)

# Cek daftar perangkat yang terlihat
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"✅ ASIK! Ditemukan {len(gpus)} GPU:")
    for gpu in gpus:
        print(f"   - {gpu.name}")
    print("\nTraining nanti otomatis pakai GPU ini.")
else:
    print("❌ GPU tidak terdeteksi oleh TensorFlow.")
    print("TensorFlow akan berjalan menggunakan CPU (Processor).")