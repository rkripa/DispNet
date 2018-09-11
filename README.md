# DispNet Draft Notes
## Disparity Estimation Network Architecture|

Here is an attempt to recreate the DispNet training environment on google cloud without the use of docker.  However using docker is the quickest way to try out inference on different networks.


## Introduction
DispNet is based on the paper by N. Mayer, E. Ilg, P. Häusser, P. Fischer, D. Cremers, A. Dosovitskiy, T. Brox, titled: 
A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation

[Paper](https://lmb.informatik.uni-freiburg.de/Publications/2016/MIFDB16/paper-MIFDB16.pdf)

[Poster](https://lmb.informatik.uni-freiburg.de/Publications/2016/MIFDB16/poster-MIFDB16.pdf)

[DispNet Project Page](https://lmb.informatik.uni-freiburg.de/Publications/2016/MIFDB16/)

[lmb-freiburg/dispnet-flownet-docker](https://github.com/lmb-freiburg/dispnet-flownet-docker)


## Data Set

Dataset used by dispnet is very large.  It took over 40 minutes for me to download and much more to untar the files.  The input data to dispnet has left and right stereo images and the output is disparity (depth) image. Disparity image is formatted as .pfm (portable float map) file.

[DispNet/FlowNet2.0 dataset subsets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)


[Sample Image](https://github.com/lmb-freiburg/dispnet-flownet-docker/blob/master/data/teaser.png)

### Scripts
- dispflownet-release/data directory contains script directory
  - download.sh
    - downloads both the flyingthings3d (cleanpass and disparity) and flyingchairs
  - make-lmdbs.sh
    - create training and test datasets in lmdb or leveldb
    - it calls:
      - ../build/tools/convert_imageset_and_disparity.bin
        - which convert_imageset_and_flow LISTFILE DB_NAME RANDOM_SHUFFLE_DATA[0 or 1] DB_BACKEND[leveldb or
 lmdb]
    - dispflownet-release/data directory contains both:
      - FlyingThings3D_release_TRAIN.list
      - FlyingThings3D_release_TEST.list
                                                          
## Environment

### Software Environment
- Ubuntu 16.04 LTS
- Cuda 8.0
- CuDnn 5.1

### Virtual Environment
- Using google cloud console, spin a VM instance with ubuntu-16.04-lts images and 1 Nvidia K80 GPU
- Peristant disk 500GB or more

### Model
- Caffe
- model
  - training
    - solver.prototxt and train.prototxt used for training
  - inference
    - deploy.tpl.prototxt  
    - weights
      - DispNet_CVPR2016.caffemodel (169MB)

## Software Drivers and Packages to Install
### CUDA
[source: google cloud]
```
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  apt-get update
  apt-get install cuda-8-0  -y
```

### CuDnn
[source: nvidia dockerfile]
```
echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

export CUDNN_VERSION=5.1.10

apt-get update && apt-get install -y --no-install-recommends \
            libcudnn5=$CUDNN_VERSION-1+cuda8.0 \
            libcudnn5-dev=$CUDNN_VERSION-1+cuda8.0 && \
    rm -rf /var/lib/apt/lists/*
```

### DispNet
[source: dispnet dockerfile]

```
apt-get update && \
    apt-get install -y --no-install-recommends \
        module-init-tools \
        build-essential \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-scipy && \
    wget --progress=bar:force:noscroll --no-check-certificate \
        https://lmb.informatik.uni-freiburg.de/data/dispflownet-release-docker.tar.gz && \
    tar xfz dispflownet-release-docker.tar.gz && \
    rm dispflownet-release-docker.tar.gz && \
    cd dispflownet-release && \
    make -j`nproc`
``` 

### Change demo.py

```
def dockerize_filepath(path):                                                 
    return os.path.join('./', path)  # /input-output --> ./
```

## Inference
- cd dispflownet-release/models/DispNet
- touch output.txt # create an empty file
- ./demo.py imgL_list.txt  imgR_list.txt  output.txt 0
  - 0 is the gpu index
  - output.txt if empty creates output in the current directory
    - else moves the output files to path specified in output.txt
  - demo.py calls caffe as:
    - bin/caffe.bin test -model tmp/deploy.prototxt -weights model/DispNet_CVPR2016.caffemodel -iterations 9 -gpu 0
    
## Train
- ./train.py 
  - train.py calls caffe as:
    - ../bin/caffe train -model ../model/train.prototxt -solver ../model/solver.prototxt
    






