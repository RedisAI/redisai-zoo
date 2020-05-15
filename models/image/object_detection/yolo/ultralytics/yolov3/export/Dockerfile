ARG REPO_DIR=/repo
ARG MODEL_FILE=model.pt

FROM ubuntu:18.04 as builder

ARG REPO_DIR
ARG MODEL_FILE

RUN apt update
RUN apt install -y python3.7 git python3-pip curl \
                   libsm6 libxext6 libxrender-dev

RUN git clone https://github.com/ultralytics/yolov3.git ${REPO_DIR}

COPY export.py ${REPO_DIR}

WORKDIR ${REPO_DIR}

RUN pip3 install numpy opencv-python matplotlib tqdm
RUN pip3 install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN python3 export.py --outfile ${MODEL_FILE}

FROM scratch as export

ARG REPO_DIR
ARG MODEL_FILE

COPY --from=builder /repo/model.pt .
