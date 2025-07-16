FROM osgeo/gdal:ubuntu-small-3.6.3

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and LaTeX packages
RUN apt-get update && \
    apt-get install -y \
        python3-pip python3-dev git curl sudo gnupg \
        latexmk \
        texlive-latex-recommended \
        texlive-fonts-recommended \
        texlive-latex-extra \
        texlive-xetex \
        texlive-font-utils \
        texlive-fonts-extra \
        fontconfig && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK and Node.js
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
        | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get update && \
    apt-get install -y nodejs google-cloud-cli && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install OpenSans LaTeX font via tlmgr
RUN tlmgr update --self && \
    tlmgr install opensans

# Install Python packages
RUN pip3 install --no-cache-dir \
    jupyterlab \
    jupytext \
    ipykernel \
    notebook \
    mystmd \
    pandas \
    numpy \
    earthengine-api \
    geemap \
    rasterio \
    folium \
    ipyleaflet \
    geopandas \
    matplotlib \
    imageio \
    tifffile \
    imagecodecs \
    seaborn \
    statsmodels