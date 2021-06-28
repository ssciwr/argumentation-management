ARG baseimage
FROM ${baseimage} AS base-system-cwb

# install dependencies
USER root
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install --no-install-recommends -y \
        python3.8 \
        python3-pip \
        autoconf \
        bison \
        flex \
        gcc \
        libc6-dev \
        libglib2.0-dev \
        libncurses5-dev \
        make \
        subversion\
    && apt-get clean \
    && apt-get -y autoremove \
    && rm -rf /var/lib/apt/lists/*

# install cwb
# copy the installation script
# should I copy to a local or tmp folder?
COPY install_cwb.sh /install_cwb.sh
# run the installer
RUN chmod +x /install_cwb.sh

# install the python dependencies and cwb-ccc
# copy the files from the git repo
COPY cwb-ccc /cwb-ccc
# install dependencies and cwb-ccc
RUN python3 /cwb-ccc/setup.py install

# install the other tools that are planned to be used
