FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# gcc/build-essential might be needed for some python libraries (like chroma/hnswlib)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Run commands
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
