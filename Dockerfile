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

# install spaCy
RUN conda install -c conda-forge spacy \
    && conda install -c \
        conda-forge spacy-lookups-data \
    && python -m spacy download en_core_web_sm \
    # && sleep 15 \
    # && python -m spacy download de_core_news_sm \
    && conda clean -a -q -y
#ENV SPACY_DIR = /home/jovyan/spacy

# install stanza
COPY docker/get_models.py /home/jovyan/.
RUN conda install -c \
        conda-forge stanza \
    && conda clean -a -q -y \
    && python get_models.py