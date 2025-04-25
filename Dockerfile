FROM rocker/ml

# Install R packages
RUN install2.r --error \
    sf \
    terra \
    rgee \
    reticulate