FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip \
    && pip install --no-cache-dir mlflow scikit-learn \
    && pip install --upgrade typing-extensions pydantic

EXPOSE 5000
CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"]
