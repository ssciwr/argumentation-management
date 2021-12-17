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
# let's try with pipenv for cwb-ccc
# RUN conda install -c conda-forge python=3.9.7 \
    # && conda install -c \
    #    conda-forge \
    #    cython \
    #    jupyter-resource-usage \
    #    pip \
    # && conda clean -a -q -y

USER jovyan
## RUN CWB_DIR=/usr/local/cwb-3.4.22 conda run -n base python -m pip install cwb-ccc
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -q pipenv \
    && pwd \
    && ls

# ENV CWB_DIR=/usr/local/cwb-3.4.22/
COPY --chown=1000:100 /docker/cwb-ccc /home/jovyan/cwb-ccc
# the below will work, have to pipenv shell and add cwb-ccc to
# pythonpath
# if not using pipenv shell: then it finds cwb-ccc in path but 
# the dependencies are of course missing
RUN cd cwb-ccc \
    && make clean \
    && make install \
    && make compile \
    && make build \
    && make test 

# install the other tools that are planned to be used
# we should probably set up conda environments so that different toolchains
# can be activated for different users
# refrain from submodules at this point until it is clear what they will use
# (varying dependencies)

#USER root

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