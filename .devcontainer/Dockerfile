FROM rocker/tidyverse:4.4.3

RUN install2.r --error \
    sf \
    terra \
    rgee \
    reticulate

RUN apt-get update && apt-get install -y python3-pip nodejs npm && apt-get clean
RUN pip3 install earthengine-api

