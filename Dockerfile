# Use a base image with Micromamba installed (for Pixi environment management)
FROM mambaorg/micromamba:1.5.0

# Set the working directory in the container
WORKDIR /app

# Copy the contents of the current directory (repo) to /app in the container
COPY . .

# Install dependencies from the pixi.lock file to ensure locked versions
RUN micromamba install -y -n base -f pixi.lock && \
    micromamba clean --all --yes

# By default, start an interactive bash shell when the container runs
CMD ["/bin/bash"]
