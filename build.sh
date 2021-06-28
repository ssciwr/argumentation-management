#!/bin/bash

# Errors are fatal
set -e

# Download cwb
svn co http://svn.code.sf.net/p/cwb/code/cwb/trunk cwb
# unfortunately, the latest revision is 3.0 while the latest
# version is 3.8 and lives in trunk - everything is bound to
# break once cwb 4.0 gets released

# prepare cwb for install - pick linux platform
sed -i 's/PLATFORM=darwin-brew/PLATFORM=linux-64/' cwb/config.mk
# this is also error-prone as the default platform seems to vary
# need to install into the standard dir for cwb-python
sed -i 's/SITE=beta-install/SITE=standard/' cwb/config.mk
