#!/bin/bash
test_dir=./test

models_dir=$test_dir/models
mkdir -p $models_dir
python -c "import stanza; stanza.download(lang='en', model_dir='${models_dir}', logging_level='info')" || echo "failed to download english model"
python -c "import stanza; stanza.download(lang='de', model_dir='${models_dir}', logging_level='info')" || echo "failed to download german model"
echo "Models downloaded to ${models_dir}."
export STANZA_TEST_HOME=$test_dir