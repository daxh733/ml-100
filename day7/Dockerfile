FROM python:3.9-slim-buster
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import nltk; nltk.download('stopwords')"
COPY models/ models/
COPY app.py .
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
