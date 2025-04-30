FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libpq-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY deployment_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r deployment_requirements.txt

# Install NLTK data
RUN python -m nltk.downloader punkt vader_lexicon stopwords wordnet

# Install spaCy model
RUN python -m spacy download en_core_web_sm

# Copy the application code
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p evidence data graph_exports

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers=4", "--timeout=120", "--reuse-port", "--reload", "main:app"]