FROM jupyter/base-notebook:python-3.9.7

USER root

# install dependencies for cwb and tools (cwb-perl, cwb-ccc)
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install --no-install-recommends -y \
        autoconf \
        bison \
        build-essential \
        cmake \
        cython3 \
        flex \
        gcc \
        git \
        less \
        libc6-dev \
        libglib2.0-0 \
        libglib2.0-dev \
        libncurses5 \
        libncurses5-dev \
        libboost-all-dev \
        libgoogle-perftools-dev \
        libpcre3-dev \
        libreadline8 \
        libreadline-dev \
        libsparsehash-dev \
        make \
        perl \
        pkg-config \
        subversion \
        wget \
    && apt-get clean \
    && apt-get -y autoremove \
    && rm -rf /var/lib/apt/lists/*

# install cwb
# COPY ./docker/cwb-3.4.32 /cwb-3.4.32
# RUN cd /cwb-3.4.32 \
# install development version from svn repo
# this unfortunately does not allow version pinning
RUN svn co http://svn.code.sf.net/p/cwb/code/cwb/trunk cwb \
    && cd cwb \
    && sed -i 's/SITE=beta-install/SITE=standard/' config.mk \
    && ./install-scripts/install-linux

# install cwb-perl for regedit
# COPY ./docker/Perl-CWB-3.0.7 /Perl-CWB-3.0.7
# RUN cd /Perl-CWB-3.0.7 \
# install development version from svn repo
# this unfortunately does not allow version pinning
RUN svn co http://svn.code.sf.net/p/cwb/code/perl/trunk cwb-perl \
    && cd cwb-perl/CWB \
    && perl Makefile.PL --config=/usr/local/bin/cwb-config \
    && make \
    && make test \
    && make install 

# install the python dependencies
RUN conda install -c conda-forge python=3.9.7 \
    && conda install -c \
       conda-forge \
       cython \
       ipywidgets \
       jupyter-resource-usage \
    && conda clean -a -q -y

# install cwb-ccc
USER jovyan
RUN conda run -n base python -m pip install cwb-ccc

# install annotator from PyPi
RUN conda run -n base python -m pip install nlpannotator

# install treetagger
RUN mkdir treetagger \
    && cd treetagger \
    && curl -sSLO http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tree-tagger-linux-3.2.4.tar.gz \
    && curl -sSLO http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tagger-scripts.tar.gz \
    && curl -sSLO http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/install-tagger.sh \
    && curl -sSLO http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/german.par.gz \
    && curl -sSLO http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/english.par.gz \
    && sh install-tagger.sh \
    && cd ..
    
# install spacy models
RUN conda install -c conda-forge spacy-lookups-data \
    && python -m spacy download en_core_web_md \
    && python -m spacy download de_core_news_md \
    # && python -m spacy download fr_core_news_md \
    # && python -m spacy download it_core_news_md \
    # && python -m spacy download ja_core_news_md \
    # && python -m spacy download pt_core_news_md \
    # && python -m spacy download ru_core_news_md \
    # && python -m spacy download es_core_news_md \
    && conda clean -a -q -y
ENV SPACY_DIR = /home/jovyan/spacy

# install stanza models
COPY docker/get_models.py /home/jovyan/.
RUN python get_models.py

# install m-giza
# RUN git clone --depth 1 --branch RELEASE-3.0 https://github.com/moses-smt/mgiza.git \
#   && cd mgiza/mgizapp \
#   && cmake . \
#   && make \
#   && make install \
#   && cd ..
# ENV MGIZA_DIR=/home/jovyan/mgiza

# install hunalign
# RUN wget ftp://ftp.mokk.bme.hu/Hunglish/src/hunalign/latest/hunalign-1.1.tgz \
#   && tar zxvf hunalign-1.1.tgz \
#   && cd hunalign-1.1/src/hunalign \
#   && make
