# Use a lightweight official Python image as the base
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK stopwords (crucial for your app)
RUN python -c "import nltk; nltk.download('stopwords')"

# ***************************************************************
# *** IMPORTANT CHANGE: Update this line to 'models/'          ***
# ***************************************************************
COPY models/ models/

# Copy your Streamlit application file into the container
COPY app.py .

# Expose the port that Streamlit runs on (default is 8501)
EXPOSE 8501

# Define the command to run your Streamlit application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]