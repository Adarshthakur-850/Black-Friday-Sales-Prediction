FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if needed (e.g. for gcc)
# RUN apt-get update && apt-get install -y build-essential

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Expose ports for API and Streamlit
EXPOSE 8000
EXPOSE 8501

# Default command: Run API
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
