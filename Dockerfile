FROM jupyter/base-notebook:python-3.9.7

USER root

# install dependencies for cwb and tools (cwb-perl, cwb-ccc)
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install --no-install-recommends -y \
        autoconf \
        bison \
        flex \
        gcc \
        libc6-dev \
        libglib2.0-0 \
        libglib2.0-dev \
        libncurses5 \
        libncurses5-dev \
        libpcre3-dev \
        make \
        subversion \
        less \
        wget \
        pkg-config \
        perl \
        libreadline8 \
        libreadline-dev \
        libboost-all-dev \
        libgoogle-perftools-dev \
        libsparsehash-dev \
        build-essential \
        cmake \
        cython3 \
    && apt-get clean \
    && apt-get -y autoremove \
    && rm -rf /var/lib/apt/lists/*

# install cwb
COPY ./docker/cwb-3.4.32 /cwb-3.4.32
RUN cd /cwb-3.4.32 \
    && sed -i 's/SITE=beta-install/SITE=standard/' config.mk \
    && ./install-scripts/install-linux

# install cwb-perl for regedit
COPY ./docker/Perl-CWB-3.0.7 /Perl-CWB-3.0.7
RUN cd /Perl-CWB-3.0.7 \
    && perl Makefile.PL --config=/usr/local/bin/cwb-config \
    && make \
    && make test \
    && make install 

# install the python dependencies and cwb-ccc
RUN conda install -c conda-forge python=3.9.7 \
    && conda install -c \
       conda-forge \
       cython \
       jupyter-resource-usage \
    && conda clean -a -q -y

USER jovyan
RUN CWB_DIR=/usr/local/cwb-3.4.22 conda run -n base python -m pip install cwb-ccc

# install the other tools that are planned to be used
# we should probably set up conda environments so that different toolchains
# can be activated for different users
# refrain from submodules at this point until it is clear what they will use
# (varying dependencies)

# install spaCy
#RUN conda install -c conda-forge spacy \
#    && conda install -c \
#        conda-forge spacy-lookups-data \
#    &&python -m spacy download en_core_web_sm\
#    && conda clean -a -q -y
#ENV SPACY_DIR = /home/jovyan/spacy

# install stanza
#RUN conda install -c \
#        conda-forge stanza \
#    && conda install -c \
#        conda-forge ipywidgets\
#    && conda clean -a -q -y#
