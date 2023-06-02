FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04

ENV TERM xterm

RUN apt update

RUN apt install -y apt-transport-https curl gnupg libgl1-mesa-glx wamerican rsync
RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
RUN mv bazel.gpg /etc/apt/trusted.gpg.d/
RUN echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list

RUN apt update

RUN apt -y install bazel htop nano cmake libncurses5-dev libncursesw5-dev git build-essential cmake python3.7 python3-pip

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip3 install --upgrade pip 

RUN pip3 install psutil gputil lz4 transformers>=4.11 tensorflow opencv-python ray==1.6.0 ray[default] ray[tune] ray[rllib]==1.6.0 tabulate numpy gym joblib tqdm tensorboardX>=1.9 pandas tree

RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /pvc/bwinter-core/medical_rl



