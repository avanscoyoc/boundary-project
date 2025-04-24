FROM rocker/tidyverse:4.4.3

# Install R packages
RUN install2.r --error \
    sf \
    terra \
    rgee \
    reticulate

# System packages + Python + Jupyter + Node
RUN apt-get update && \
    apt-get install -y \
        python3-pip \
        nodejs \
        npm \
        curl \
        sudo \
        wget \
        gdebi-core && \
    apt-get clean

# Python packages
RUN pip3 install --no-cache-dir earthengine-api jupyterlab

# Install RStudio Server (optional, port 8787)
RUN wget https://download2.rstudio.org/server/bionic/amd64/rstudio-server-2023.06.2-561-amd64.deb && \
    gdebi -n rstudio-server-2023.06.2-561-amd64.deb && \
    rm rstudio-server-2023.06.2-561-amd64.deb

# Create non-root user (needed for RStudio and VS Code Server)
RUN useradd -m devuser && echo "devuser:devpass" | chpasswd && adduser devuser sudo

# Install code-server (optional VS Code in browser, port 8080)
RUN curl -fsSL https://code-server.dev/install.sh | sh

# Expose necessary ports
EXPOSE 8888 8787 8080

# Default command: Launch JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root"]

