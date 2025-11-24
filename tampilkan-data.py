import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pathlib 

# Akses ke dataset
data_dir = pathlib.Path("D:/Kuliah/Machine Learning/Project/archive (3)/dataset_itr2/train")
val_dir = pathlib.Path("D:/Kuliah/Machine Learning/Project/archive (3)/dataset_itr2/test")

class_names = [item.name for item in data_dir.glob('*') if item.is_dir()]
class_names.sort()

print(f"Ditemukan {len(class_names)} Kelas: {class_names}\n")
print(" Detail Jumlah Gambar ")

total_train = 0
counts = [] 

for kelas in class_names:
    folder_kelas = list(data_dir.glob(f'{kelas}/*'))
    
    jumlah = len(folder_kelas)
    counts.append(jumlah)
    total_train += jumlah
    
    print(f"- {kelas:<25} : {jumlah} gambar \n")

val_count = len(list(val_dir.glob('*/*.jpg')))

print(f"Total Data Validation: {val_count} gambar")
print(f"Total Data Train        : {total_train} gambar")


# Buat Grafik
plt.figure(figsize=(10, 6))
bars = plt.bar(class_names, counts, color=['#ff9999','#66b3ff','#99ff99','#ffcc99'])

plt.xlabel('Nama Kelas Penyakit')
plt.ylabel('Jumlah Gambar')
plt.title('Distribusi Data Training Daun Apel')
plt.xticks(rotation=15) 
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', va='bottom')

plt.show()
