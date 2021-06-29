FROM jupyter/base-notebook

USER root
# Create a non-root user
ARG username=inga-temp
ARG uid=1100
ARG gid=100
ENV USER $username
ENV UID $uid
ENV GID $gid
ENV HOME /home/$USER
RUN adduser --disabled-password \
   --gecos "Non-root user" \
   --uid $UID \
   --gid $GID \
   --home $HOME \
   $USER

# install dependencies
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install --no-install-recommends -y \
        python3.8 \
        python3-pip \
        python3-dev \
        autoconf \
        bison \
        flex \
        gcc \
        libc6-dev \
        libglib2.0-dev \
        libncurses5-dev \
        make \
        wget \
        libreadline-dev \
    && apt-get clean \
    && apt-get -y autoremove \
    && rm -rf /var/lib/apt/lists/*

# install cwb
RUN wget https://downloads.sourceforge.net/project/cwb/cwb/cwb-3.4-beta/cwb-3.4.22-source.tar.gz \
    && tar xzvf cwb-3.4.22-source.tar.gz \
    && rm cwb-3.4.22-source.tar.gz \
    && cd cwb-3.4.22 \
    && make clean \
    && make all PLATFORM=linux SITE=standard \
    && make install \
    && pwd \
    && ls .. \
    && ls /usr/local

# install the python dependencies and cwb-ccc
USER inga-temp
# RUN python3 -m pip install cython
RUN CWB_DIR=/usr/local/cwb-3.4.22 python3 -m pip install cwb-ccc

# install the other tools that are planned to be used
# we should probably set up conda environments so that different toolboxes
# can be activated for different users
