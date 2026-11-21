FROM python:3.10-slim

WORKDIR /app

# OpenCV (headless) ve derleme için gerekli sistem bağımlılıkları
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıklarını yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyalarını ve eğitilmiş modelleri kopyala
COPY models/ ./models/
COPY meta_data.xlsx .
COPY app.py .

# Varsayılan port
EXPOSE 7860

# Hem Render ($PORT) hem Hugging Face (7860) ile uyumlu başlatma komutu
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]