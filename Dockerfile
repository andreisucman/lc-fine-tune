FROM python:3.11-slim

# Install dependencies to build Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .
COPY main.py .

# Install required packages
RUN pip install --upgrade pip setuptools wheel packaging torch \
    && pip install flash-attn --no-build-isolation \
    && pip install -r requirements.txt

# Run the app
CMD ["python", "main.py"]
