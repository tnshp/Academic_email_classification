# Use the official PyTorch image with CUDA 11.8 as the base image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set the working directory in the container to the current directory
WORKDIR /

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code (including app.py) into the container
COPY . .

# Command to run the application
CMD ["python", "app.py", "examples/Q1.txt", "--output_file", "examples/A1.txt"]
