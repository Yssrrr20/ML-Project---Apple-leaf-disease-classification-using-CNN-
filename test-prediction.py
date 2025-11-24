################################### 
# Ini Buat test data dari folder validation
###################################

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 1. Setup Path
val_dir = pathlib.Path("D:/Kuliah/Machine Learning/Project/archive (3)/dataset_itr2/test")
BATCH_SIZE = 32
IMG_SIZE = (128, 128)

print("--- Memuat Data Validasi (Acak) ---")

# 2. Muat Data dengan SHUFFLE=TRUE  
val_ds_shuffled = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    shuffle=True,  
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Ambil nama kelas
class_names = val_ds_shuffled.class_names
clean_class_names = [name.replace("Apple___", "").replace("_", " ") for name in class_names]

# 3. Load Model
model = tf.keras.models.load_model('model_daun_apel.h5')
print("✅ Model loaded.")

# 4. Ambil 1 Batch (32 Gambar) yang sudah diacak
image_batch, label_batch = next(iter(val_ds_shuffled))

# Prediksi
predictions = model.predict(image_batch)

# 5. Visualisasi 9 Gambar Campur
plt.figure(figsize=(12, 12))

for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    
    # Tampilkan Gambar
    plt.imshow(image_batch[i].numpy().astype("uint8"))
    
    # Ambil Prediksi
    predicted_index = np.argmax(predictions[i])
    confidence = np.max(predictions[i]) * 100
    
    predicted_label = clean_class_names[predicted_index]
    true_label = clean_class_names[label_batch[i]]
    
    # Logika Warna
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    
    plt.title(f"Pred: {predicted_label} ({confidence:.1f}%)\nTrue: {true_label}", 
              color=color, fontsize=10)
    plt.axis("off")

plt.tight_layout()
plt.show()