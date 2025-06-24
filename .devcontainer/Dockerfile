FROM osgeo/gdal:ubuntu-small-3.6.3

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

# Install Python, pip, PostgreSQL server, PostGIS, and utilities
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    postgresql postgresql-contrib \
    postgis \
    git \
    curl \
    sudo

# Upgrade pip and install Python packages
RUN pip3 install --no-cache-dir \
    jupyterlab \
    ipykernel \
    notebook \
    pandas \
    numpy \
    earthengine-api \
    geemap \
    folium \
    ipyleaflet \
    geopandas \
    sqlalchemy psycopg2-binary

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-cli -y
    
# Copy startup script into container
COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

# Expose PostgreSQL port (optional, for external access)
EXPOSE 5432

CMD ["/usr/local/bin/start.sh"]