# Use official Python image
FROM python:3.11-slim

# Create working directory
WORKDIR /app

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY new_requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r new_requirements.txt

# Copy files into container
COPY . /app

# Download and link spaCy model manually (this avoids runtime loading issues)
RUN python -m spacy download en_core_web_sm
RUN python -m spacy link en_core_web_sm en_core_web_sm

# Expose FastAPI port
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "app_copy:app", "--host", "0.0.0.0", "--port", "8000"]
