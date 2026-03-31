# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory to /app
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app with uvicorn
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
