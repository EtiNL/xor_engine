# Use the official CUDA base image
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Set working directory
WORKDIR /app

# Copy application files
COPY . .

# Install dependencies and set locale
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    libx11-dev \
    libgl1-mesa-dev \
    pkg-config \
    software-properties-common \
    xdg-utils \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    x11-apps \
    locales \
    desktop-file-utils \
    feh \
    imagemagick \
    xdotool \
    libsdl2-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install nvcc
RUN apt-get update && apt-get install -y \
    cuda-toolkit-12-4 \
    && rm -rf /var/lib/apt/lists/*

# Copy and compile CUDA kernel
RUN mkdir -p src/gpu_utils
COPY src/gpu_utils/kernel.cu src/gpu_utils/kernel.cu
RUN nvcc -ptx -o src/gpu_utils/kernel.ptx src/gpu_utils/kernel.cu

# Create the input group if it doesn't exist and add root user to it
RUN groupadd -f input && usermod -aG input root

# Set environment variable for X11
ENV DISPLAY=host.docker.internal:0.0

# Set the default command to run the Rust application
CMD ["cargo", "run"]
