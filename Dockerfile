FROM jupyter/datascience-notebook:latest

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
