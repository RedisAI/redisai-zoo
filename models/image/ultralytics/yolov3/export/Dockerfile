ARG filename=model.pt
ARG device=cpu

ARG REPO_DIR=/repo

FROM ubuntu:18.04 as builder

ARG filename
ARG device

ARG REPO_DIR

RUN apt update
RUN apt install -y python3.7 git python3-pip curl \
                   libsm6 libxext6 libxrender-dev

RUN git clone https://github.com/ultralytics/yolov3.git ${REPO_DIR}

COPY export.py ${REPO_DIR}

WORKDIR ${REPO_DIR}

RUN pip3 install numpy opencv-python matplotlib tqdm
RUN pip3 install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN python3 export.py --device ${device} --filename ${filename}

FROM scratch as export

ARG filename
ARG device

ARG REPO_DIR

COPY --from=builder ${REPO_DIR}/${filename} .
