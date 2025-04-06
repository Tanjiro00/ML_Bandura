# Use the official Python image from the Docker Hub
FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Копируем файл requirements.txt в контейнер по пути /app
COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        build-essential \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Устанавливаем зависимости, указанные в requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

# Expose the port that the app runs on
EXPOSE 8009

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8009"]