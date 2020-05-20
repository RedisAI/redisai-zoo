ARG REPO_DIR=/repo
ARG MODEL_FILE=model.pt

FROM ubuntu:18.04 as builder

ARG REPO_DIR
ARG MODEL_FILE

RUN apt update
RUN apt install -y python3.7 git python3-pip curl \
                   libsm6 libxext6 libxrender-dev

COPY predict.py ${REPO_DIR}

WORKDIR ${REPO_DIR}

RUN pip3 install -r requirements.txt

FROM scratch as predict

ARG REPO_DIR
ARG MODEL_FILE

RUN python3 predict.py --outfile ${MODEL_FILE}

