import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pathlib
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pathlib 
import numpy as np

# ambil dataset
train_path = pathlib.Path("D:/Kuliah/Machine Learning/Project/archive (3)/dataset_itr2/train")
val_path = pathlib.Path("D:/Kuliah/Machine Learning/Project/archive (3)/dataset_itr2/test")

train_dir = pathlib.Path(train_path)
val_dir = pathlib.Path(val_path)

# Konfigurasi
BATCH_SIZE = 32           
IMG_SIZE   = (128, 128)   

print("\n--- Sedang Memuat Data ---")

# Muat Data Training
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    shuffle=True,         
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Muat Data Validasi
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    shuffle=False,        
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# clean class name 
raw_class_names = train_ds.class_names

class_names = []
for name in raw_class_names:
    clean_name = name.replace("Apple___", "").replace("_", " ")
    class_names.append(clean_name)

print(f"class : {class_names}")

# Optimasi Performa
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("\n--- Arsitektur Model ---")

num_classes = len(class_names)

# Augmentasi Data
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"), 
  layers.RandomRotation(0.2),                   
])

# Susunan Layer CNN 
model = models.Sequential([
  # 1. Input Layer
  layers.Input(shape=(128, 128, 3)),
  
  # 2. Preprocessing
  data_augmentation,        
  layers.Rescaling(1./255), 

  # 3. Feature Extraction
  # Blok 1 
  layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
  layers.MaxPooling2D((2, 2)),
  
  # Blok 2
  layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
  layers.MaxPooling2D((2, 2)),
  
  # Blok 3
  layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
  layers.MaxPooling2D((2, 2)),

  # 4. Classification
  layers.Flatten(),                   
  layers.Dense(64, activation='relu'),
  layers.Dropout(0.2),                
  
  # 5. Output Layer
  layers.Dense(num_classes, activation='softmax')
])

model.summary()

# Compile dan Training

# 1. Compile 
model.compile(
    optimizer='adam', 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
    metrics=['accuracy'] #
)

print("\n MULAI TRAINING ")

# 2. Start Training
EPOCHS = 15 
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Chart dan save model
model.save('model_daun_apel.h5')
print("\n Model berhasil disimpan sebagai 'model_daun_apel.h5'")

# 4. Tampilkan Grafik
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))

# Grafik Akurasi
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Grafik Akurasi')

# Grafik Error (Loss)
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Grafik Error (Loss)')

plt.show() 