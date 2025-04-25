FROM python:3.9-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    jupyterlab \
    ipykernel \
    notebook \
    earthengine-api \
    geemap \
    folium \
    ipyleaflet \
    pandas \
    numpy

CMD ["bash"]