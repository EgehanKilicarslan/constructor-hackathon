# Use a modern Python slim image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies. 'build-essential' is good practice for Python projects
# that might have C extensions, which 'generals-bots' or its dependencies could have.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python application code
COPY main.py .

# Copy optional .env files (using wildcard to prevent build failure if missing)
COPY .env* ./

# Command to run the application
CMD ["python", "main.py"]