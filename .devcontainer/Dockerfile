FROM osgeo/gdal:ubuntu-small-3.6.3

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3-pip python3-dev git curl sudo gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
      | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update && \
    apt-get install -y google-cloud-cli && \
    apt-get clean && rm -rf /var/lib/apt/lists/*v

RUN pip3 install --no-cache-dir \
    jupyterlab \
    jupytext \
    ipykernel \
    notebook \
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
    imagecodecs 