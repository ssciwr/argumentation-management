FROM jupyter/base-notebook

USER root
# Create a non-root user
# ARG username=inga-temp
# ARG uid=1100
# ARG gid=100
# ENV USER $username
# ENV UID $uid
# ENV GID $gid
# ENV HOME /home/$USER
# RUN adduser --disabled-password \
#   --gecos "Non-root user" \
#   --uid $UID \
#   --gid $GID \
#   --home $HOME \
#   $USER

# install dependencies for cwb
# I'd like to keep this separate from the python part
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install --no-install-recommends -y \
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
    && ls /usr/local/cwb-3.4.22

# install the python dependencies and cwb-ccc
# stay with conda for consistency
RUN conda install -c conda-forge python=3.6 && \
    conda install -c conda-forge \
        cython \
    && conda clean -a -q -y
# USER inga-temp
USER jovyan
# or should I use jovyan? root?
RUN CWB_DIR=/usr/local/cwb-3.4.22 conda run -n base python -m pip install cwb-python

# install the other tools that are planned to be used
# we should probably set up conda environments so that different toolboxes
# can be activated for different users
