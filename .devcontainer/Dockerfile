# Dev container: Python pipeline (scripts 01–04) + R statistics (scripts 05–10).
# Builds natively on linux/amd64 (Windows with WSL2, Intel Macs, Linux) and linux/arm64 (Apple Silicon).
# Build context is the repo root (devcontainer.json); .dockerignore limits it to the files copied below.

# Pinned by digest, so every rebuild starts from the same Ubuntu 22.04 + GDAL 3.6.3 image.
FROM osgeo/gdal:ubuntu-small-3.6.3@sha256:398d5aca0ca88295c13ada87e0f382ed409dac13a952461187be72ce3c376513

WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive

# System tools. google-cloud-cli stays: ee.Authenticate() signs in through gcloud when it is installed.
RUN apt-get update && \
    apt-get install -y python3-pip git curl ca-certificates gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
      > /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update && \
    apt-get install -y google-cloud-cli && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Python: packages the code imports (requirements.txt), with every dependency pinned (constraints.txt).
COPY requirements.txt constraints.txt /tmp/python/
RUN pip3 install --no-cache-dir -r /tmp/python/requirements.txt -c /tmp/python/constraints.txt && \
    rm -rf /tmp/python

# R: an isolated conda-forge environment managed by pixi at /opt/r (.devcontainer/pixi.toml + pixi.lock).
# Nothing goes into /usr, so the system GDAL and Python are untouched. `R` and `Rscript` on PATH are
# small wrappers that activate the environment first (sf needs this to find its PROJ database).
ARG PIXI_VERSION=0.84.1
RUN curl -fsSL "https://github.com/prefix-dev/pixi/releases/download/v${PIXI_VERSION}/pixi-$(uname -m)-unknown-linux-musl.tar.gz" \
      | tar -xz -C /usr/local/bin pixi
COPY .devcontainer/pixi.toml .devcontainer/pixi.lock /opt/r/
RUN cd /opt/r && \
    pixi install --locked && \
    pixi run --locked install-cran && \
    pixi shell-hook --locked -s bash > /opt/r/activate.sh && \
    for b in R Rscript; do \
      printf '#!/bin/bash\nsource /opt/r/activate.sh\nexec /opt/r/.pixi/envs/default/bin/%s "$@"\n' "$b" > /usr/local/bin/$b && \
      chmod +x /usr/local/bin/$b; \
    done && \
    pixi clean cache --yes && \
    Rscript -e 'for (p in c("sf", "nlme", "emmeans", "spdep", "zyp", "testthat")) library(p, character.only = TRUE); cat(R.version.string, "\n")'
