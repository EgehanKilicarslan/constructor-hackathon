# Use a modern Python base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main application script
COPY main.py .env* ./

# Set the entrypoint for the application
# This allows the script to be run directly with arguments
ENTRYPOINT ["python", "main.py"]

# Define a default command if no arguments are provided
# This will run the 'evaluate' function with default parameters
CMD ["evaluate"]