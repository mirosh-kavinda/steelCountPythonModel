# Use an official TensorFlow GPU runtime as a parent image
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install required system libraries
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0

# Install NVIDIA Container Toolkit
RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - \
    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
       tee /etc/apt/sources.list.d/nvidia-docker.list

# Install blinker without uninstalling
RUN python -m pip install --ignore-installed blinker==1.4

# Install any needed packages specified in requirements.txt
RUN pip install -U Flask numpy opencv-python scikit-image 
RUN apt-get update && apt-get install -y nvidia-docker2 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the environment variable to use the GPU
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility



# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME countmein

# Run app.py when the container launches
CMD ["python", "app.py"]
