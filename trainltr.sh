#!/usr/bin/env bash

# Get set the ranklib path, and build the jar if it does not exist
ranklib_path=./ranklib/build/libs/ranklib.jar
if [ ! -f ${ranklib_path} ]; then
    cd ./ranklib/
    ./gradlew build
    cd ../
fi

# Create training data by using elasticsearch to generate features for RankLib
python3 ./ltrfeatures.py --mapping ./pmid-mapping.json --queries ./elastic-pico-queries.json --output training.txt

# Train a model
# https://sourceforge.net/p/lemur/wiki/RankLib%20How%20to%20use/
java -jar ${ranklib_path} -train training.txt -ranker 6 -save model.txt -gmax 1 -metric2t NDCG@100 -tvs 0.2
