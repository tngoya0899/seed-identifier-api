FROM python:3.10-slim

WORKDIR /app

# Gerekli sistem bağımlılıkları
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Bağımlılıkları yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyalarını ve modelleri kopyala
COPY models/ ./models/
COPY meta_data.xlsx .
COPY app.py .

# Hugging Face Spaces portu (7860 standarttır)
EXPOSE 7860

# FastAPI uygulamasını başlat
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]