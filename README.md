## Cara Menjalankan Project

### 1. Clone Repository
Unduh source code ke komputer Anda:
```bash
git clone [https://github.com/Yssrrr20/ML-Project---Apple-leaf-disease-classification-using-CNN-.git](https://github.com/Yssrrr20/ML-Project---Apple-leaf-disease-classification-using-CNN-.git)
cd ML-Project---Apple-leaf-disease-classification-using-CNN-

# Untuk Windows
python -m venv env_apel

# Untuk Mac/Linux
python3 -m venv env_apel

# Aktifkan venv
env_apel\Scripts\activate
atau
env_apel\Scripts\activate

# Install Library
pip install -r requirements.txt

# Jalankan Program dan tes model dari validation folder
python test_prediksi.py

# Jalankan Program dan tes model dari gambar yang di download ke folder upload gambar
python prediction-upload.py

# Kalau mau coba training 
python train_model.py
```
### Note
Perhatikan file path 


