FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Set working directory
WORKDIR /app

# Copy application files
COPY . .

# Install dependencies
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
    && rm -rf /var/lib/apt/lists/*

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

# Copy the shell script and set execute permission
COPY run.sh .
RUN chmod +x run.sh

# Set the command to run the shell script
CMD ["./run.sh"]
