FROM jupyter/scipy-notebook:python-3.9

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install jupyterlab-code-formatter \
    jupyterlab_vscode \
    && jupyter lab build
