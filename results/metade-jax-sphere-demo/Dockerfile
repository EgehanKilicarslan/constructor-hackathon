# Use a modern Python slim image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install build dependencies for git packages and JAX
# JAX requires some system packages for its wheels or compilation
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install JAX (CPU version for slim image compatibility)
# The original README suggests `pip install -U jax` for CPU.
RUN pip install --no-cache-dir "jax[cpu]"

# Install evox and metade from git as specified in their README
# Use a specific version for evox as recommended
RUN pip install --no-cache-dir git+https://github.com/EMI-Group/evox.git@v1.1.1
RUN pip install --no-cache-dir git+https://github.com/EMI-Group/metade.git

# Copy the application code
COPY main.py .env* ./

# Command to run the application
CMD ["python", "main.py"]
