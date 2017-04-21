#!/usr/bin/env bash

# These variables are placed at the top of this file for convenience for you to edit
ranklib_path=./ranklib/build/libs/ranklib.jar # Path to RankLib
model_name=model # Model name exported from RankLib _and_ used in elasticsearch
elasticsearch_host=localhost
elasticsearch_port=9200

# Check if elasticsearch is running before proceeding
nc -zc ${elasticsearch_host} ${elasticsearch_port} &> /dev/null
if [ $? -ne 0 ]
then
    echo 'elasticsearch is not running, or it is running on a different address than ${elasticsearch_host}:${elasticsearch_port}.'
    exit 1
fi

# Get set the ranklib path, and build the jar if it does not exist
if [ ! -f ${ranklib_path} ]; then
    cd ./ranklib/
    ./gradlew build
    cd ../
fi

# Create training data by using elasticsearch to generate features for RankLib
python3 ./ltrfeatures.py --mapping ./pmid-mapping.json --queries ./elastic-pico-queries.json --output training.txt

# Train a model
# https://sourceforge.net/p/lemur/wiki/RankLib%20How%20to%20use/
java -jar ${ranklib_path} -train training.txt -ranker 6 -save ${model_name}.txt -gmax 1 -metric2t NDCG@100 -tvs 0.2

# Now we can upload the model to elasticsearch
python3 ./uploadscript.py --input ${model_name}.txt --elastic-url http://${elasticsearch_host}:${elasticsearch_port} -v