#!/bin/bash

# Errors are fatal
set -e

# install cwb
# it is important to install as sudo as otherwise cwb-ccc
# will not install correctly (needs directory structure)
# not sure if the user=root from the dockerfile persists down
# to this level
cd cwb
make clean
make depend
make cl
make editline
make utils
make cqp
# sudo make install
make install
