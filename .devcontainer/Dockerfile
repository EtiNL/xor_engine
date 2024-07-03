# Use the official CUDA base image
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Set working directory
WORKDIR /app

# Copy application files
COPY . .

# Install dependencies and set locale
RUN apt-get update && apt-get install -y \
    build-essential \
    libsdl2-dev \
    libwayland-dev \
    locales \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set environment variables for Wayland
ENV WAYLAND_DISPLAY=wayland-0
ENV XDG_RUNTIME_DIR=/run/user/1000

# Set the default command to run the Rust application
CMD ["cargo", "run"]
