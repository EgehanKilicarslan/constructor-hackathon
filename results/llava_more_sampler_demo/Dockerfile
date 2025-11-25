# Use a modern Python base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main application script
COPY main.py .env* ./

# Command to run the application
CMD ["python", "main.py"]
