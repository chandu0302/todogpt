FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y postgresql-client && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the port langgraph dev actually listens on (see CMD below)
EXPOSE 2024

# Start LangGraph API server
# The langgraph CLI automatically reads langgraph.json
CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "2024"]