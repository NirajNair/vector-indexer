FROM rust:latest

# Set working directory
WORKDIR /workspace

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for better caching
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY . .

# Default command (can be overridden)
CMD ["bash"]

