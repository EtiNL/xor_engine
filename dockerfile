# Use the official CUDA base image
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Set working directory
WORKDIR /app

# Copy application files
COPY . .

# Set DEBIAN_FRONTEND to noninteractive to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies and set locale
RUN apt-get update && apt-get install -y \
    build-essential \
    libsdl2-dev \
    libsdl2-ttf-dev \
    libwayland-dev \
    locales \
    curl \
    wget \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxkbcommon-x11-0 \
    libxcb-xinput0 \
    libxcb-cursor0 \
    libnss3 \
    libxcomposite1 \
    libxdamage1 \
    libxtst6 \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA Toolkit
RUN apt-get update && apt-get install -y cuda

# Set up locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Nsight Systems
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nsight-systems-2024.2.3_2024.2.3.38-1_amd64.deb && \
    apt-get install -y ./nsight-systems-2024.2.3_2024.2.3.38-1_amd64.deb && \
    rm nsight-systems-2024.2.3_2024.2.3.38-1_amd64.deb

# Set environment variables for Wayland
ENV WAYLAND_DISPLAY=wayland-0
ENV XDG_RUNTIME_DIR=/run/user/1000

# Set the default command to run the Rust application
CMD ["cargo", "run"]
