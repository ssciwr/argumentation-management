FROM jupyter/base-notebook

USER root
# Create a non-root user
# there is already jovyan, so no need for this
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

# install dependencies for cwb, and tools (moses, fastalign)
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
    && make install

# install the python dependencies and cwb-ccc
# stay with conda for consistency
RUN conda install -c conda-forge python=3.6 \
    && conda install -c \
       conda-forge \
       cython \
    && conda clean -a -q -y
USER jovyan
RUN CWB_DIR=/usr/local/cwb-3.4.22 conda run -n base python -m pip install cwb-ccc
# RUN CWB_DIR=/usr/local/cwb-3.4.22 conda run -n base python -m pip install cwb-python

# install the other tools that are planned to be used
# we should probably set up conda environments so that different toolchains
# can be activated for different users

# install m-giza
RUN git clone --depth 1 --branch RELEASE-3.0 https://github.com/moses-smt/mgiza.git \
   && cd mgiza/mgizapp \
   && cmake . \
   && make \
   && make install \
   && cd ..
ENV MGIZA_DIR=/home/jovyan/mgiza
# not sure if these environment variables persist with the correct directory

# install fastalign - there are no versions/branches unfortunately
RUN git clone --depth 1 https://github.com/clab/fast_align.git \
   && cd fast_align \
   && mkdir build \
   && cd build \
   && cmake .. \
   && make \
   && cd ../..
ENV FASTALIGN_DIR=/home/jovyan/fast_align
# not sure if these environment variables persist with the correct directory

# install moses - this takes ages unfortunately
# not very happy with the manual boost path
# also not very happy with the ancient version of moses
# commenting this out for now as I am not sure if they need moses - mgiza does not need it apparently
# RUN git clone --depth 1 --branch RELEASE-4.0 https://github.com/moses-smt/mosesdecoder.git \
#    && cd mosesdecoder \
#    && ./bjam --prefix=/usr/lib/x86_64-linux-gnu -j4
# ENV MOSES_DIR=/home/jovyan/mosesdecoder

## install alignment-scripts - there are no versions/branches unfortunately
RUN git clone --depth 1 https://github.com/lilt/alignment-scripts.git

# get sample corpus
USER root
# RUN wget http://cwb.sourceforge.net/temp/DemoCorpus-German-1.0.tar.gz \
#       && tar xzvf DemoCorpus-German-1.0.tar.gz \
#       && rm DemoCorpus-German-1.0.tar.gz
ADD http://cwb.sourceforge.net/temp/DemoCorpus-German-1.0.tar.gz /usr/local/cwb-3.4.22/
RUN cd /usr/local/cwb-3.4.22/ \
        && tar xzvf DemoCorpus-German-1.0.tar.gz \
        && rm DemoCorpus-German-1.0.tar.gz
