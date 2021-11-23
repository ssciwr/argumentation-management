FROM jupyter/base-notebook:584f43f06586

USER root
# Create a non-root user
# there is already jovyan, so no need for this

# install dependencies for cwb and tools (moses, fastalign)
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
        git \
        libboost-all-dev \
        libgoogle-perftools-dev \
        libsparsehash-dev \
        build-essential \
        cmake \
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
    && cd .. \
    && rm -rf cwb-3.4.22

# install the python dependencies and cwb-ccc
# stay with conda for consistency
RUN conda install -c conda-forge python=3.6 \
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

USER root

# install spaCy
RUN conda install -c conda-forge spacy \
    && conda install -c \
        conda-forge spacy-lookups-data \
    &&python -m spacy download en_core_web_sm\
    && conda clean -a -q -y
ENV SPACY_DIR = /home/jovyan/spacy

# install stanza
RUN conda install -c \
        conda-forge stanza \
    && conda install -c \
        conda-forge ipywidgets\
    && conda clean -a -q -y

# install m-giza
#RUN git clone --depth 1 --branch RELEASE-3.0 https://github.com/moses-smt/mgiza.git \
#   && cd mgiza/mgizapp \
#   && cmake . \
#   && make \
#   && make install \
#   && cd ..
#ENV MGIZA_DIR=/home/jovyan/mgiza

# install fastalign - there are no versions/branches unfortunately
#RUN git clone --depth 1 https://github.com/clab/fast_align.git \
#   && cd fast_align \
#   && mkdir build \
#   && cd build \
#   && cmake .. \
#   && make \
#   && cd ../..
#ENV FASTALIGN_DIR=/home/jovyan/fast_align

# install moses - this takes ages unfortunately
# not very happy with the manual boost path
# also not very happy with the ancient version of moses
# commenting this out for now as I am not sure if they need moses - mgiza does not need it apparently
# RUN git clone --depth 1 --branch RELEASE-4.0 https://github.com/moses-smt/mosesdecoder.git \
#    && cd mosesdecoder \
#    && ./bjam --prefix=/usr/lib/x86_64-linux-gnu -j4
# ENV MOSES_DIR=/home/jovyan/mosesdecoder

## install alignment-scripts - there are no versions/branches unfortunately
#RUN git clone --depth 1 https://github.com/lilt/alignment-scripts.git