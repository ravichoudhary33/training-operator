## dockerfile for multiworker strategy for tensorflow

FROM tensorflow/tensorflow:2.1.0-gpu-py3

RUN pip install tensorflow_datasets==2.1.0

COPY multi_worker_strategy-with-keras.py /
COPY data/x_train.npy /
COPY data/x_test.npy /
COPY data/y_train.npy /
COPY data/y_test.npy /
COPY ./saved_model/ /train/saved_model/

ENTRYPOINT ["python", "/multi_worker_strategy-with-keras.py", "--saved_model_dir", "/train/saved_model/", "--checkpoint_dir", "/train/checkpoint"]


## dockerfile for distributed training of tensorflow model with mpi horodov

# FROM horovod/horovod:0.20.0-tf2.3.0-torch1.6.0-mxnet1.6.0.post0-py3.7-cuda10.1
# # FROM horovod/horovod:latest

# # Allow OpenSSH to talk to containers without asking for confirmation
# # by disabling StrictHostKeyChecking.
# # mpi-operator mounts the .ssh folder from a Secret. For that to work, we need
# # to disable UserKnownHostsFile to avoid write permissions.
# # Disabling StrictModes avoids directory and files read permission checks.
# RUN sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
#     echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
#     sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

# # # dockerfile for horovod from scratch
# # FROM ubuntu:18.04

# # ENV TENSORFLOW_VERSION=2.3.0
# # ENV PYTORCH_VERSION=1.6.0
# # ENV TORCHVISION_VERSION=0.7.0
# # ENV MXNET_VERSION=1.6.0.post0

# # # Python 3.7 is supported by Ubuntu Bionic out of the box
# # ARG python=3.7
# # ENV PYTHON_VERSION=${python}

# # # Set default shell to /bin/bash
# # SHELL ["/bin/bash", "-cu"]

# # RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
# #         build-essential \
# #         cmake \
# #         g++-4.8 \
# #         git \
# #         curl \
# #         vim \
# #         wget \
# #         ca-certificates \
# #         libjpeg-dev \
# #         libpng-dev \
# #         python${PYTHON_VERSION} \
# #         python${PYTHON_VERSION}-dev \
# #         python${PYTHON_VERSION}-distutils \
# #         librdmacm1 \
# #         libibverbs1 \
# #         ibverbs-providers

# # RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# # RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
# #     python get-pip.py && \
# #     rm get-pip.py

# # # Install TensorFlow, Keras, PyTorch and MXNet
# # RUN pip install future typing
# # RUN pip install tensorflow-cpu==${TENSORFLOW_VERSION} \
# #                 keras \
# #                 h5py
# # # RUN pip install torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION}
# # # RUN pip install mxnet==${MXNET_VERSION}

# # # Install Open MPI
# # RUN mkdir /tmp/openmpi && \
# #     cd /tmp/openmpi && \
# #     wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
# #     tar zxf openmpi-4.0.0.tar.gz && \
# #     cd openmpi-4.0.0 && \
# #     ./configure --enable-orterun-prefix-by-default && \
# #     make -j $(nproc) all && \
# #     make install && \
# #     ldconfig && \
# #     rm -rf /tmp/openmpi

# # # Install Horovod
# # # RUN HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MXNET=1 \
# # #     pip install --no-cache-dir horovod

# # RUN HOROVOD_WITH_TENSORFLOW=1 \
# #     pip install --no-cache-dir horovod

# # # Install OpenSSH for MPI to communicate between containers
# # RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
# #     mkdir -p /var/run/sshd

# # # Allow OpenSSH to talk to containers without asking for confirmation
# # RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
# #     echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
# #     mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config


# RUN pip install tensorflow_datasets==2.1.0

# COPY tensorflow_mnist_mpi-operator.py /
# COPY data/x_train.npy /
# COPY data/x_test.npy /
# COPY data/y_train.npy /
# COPY data/y_test.npy /

# ENTRYPOINT ["mpirun",\
#     "-np", "2", \
#     "--allow-run-as-root",\
#     "-bind-to", "none",\
#     "-map-by", "slot",\
#     "-x", "LD_LIBRARY_PATH",\
#     "-x", "PATH",\
#     "-mca", "pml", "ob1",\
#     "-mca", "btl", "^openib",\
#     "-mca", "routed", "direct",\
#     "python", "/tensorflow_mnist_mpi-operator.py",\
#     "--saved_model_dir","/train/saved_model/", \
#     "--checkpoint_dir","/train/checkpoint"\
# ]

# # ENTRYPOINT ["python", "/tensorflow_mnist_mpi-operator.py", "--saved_model_dir", "/train/saved_model/", "--checkpoint_dir", "/train/checkpoint"]