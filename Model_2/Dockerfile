FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy app files
COPY . /app

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
