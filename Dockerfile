# Use an official Python runtime as the base image
FROM ubuntu:16.04

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python \
    python-pip \
    python-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git

# Install TensorFlow and other Python dependencies
# RUN pip install tensorflow-gpu==1.14.0
RUN pip install scikit-image==0.14.0
RUN pip install opencv-python==3.4.2.17
RUN pip install numpy==1.14.5

# Copy your application code to the container
COPY . /app

# Define the command to run your application
CMD [ "python", "countmein_main.py" ]