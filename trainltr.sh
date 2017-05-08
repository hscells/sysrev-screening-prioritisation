#!/usr/bin/env bash

# These variables are placed at the top of this file for convenience for you to edit
ranklib_path=./RankLib-2.8.jar # Path to RankLib
model_name=model # Model name exported from RankLib _and_ used in elasticsearch
elasticsearch_host=localhost
elasticsearch_port=9200
elasticsearch_index=med
queries_json_file=./elastic-pico-queries.json
output_dest=./results/

# Don't touch these variable
elasticsearch_address=${elasticsearch_host}:${elasticsearch_port} # calculated elasticsearch url
timestamp=$(date | sed 's/ //g' | cut -c 1-14) # unique name for the file
trec_baseline=${output_dest}trec_baseline_${timestamp}.txt
trec_ltr=${output_dest}trec_ltr_${timestamp}.txt

function log {
    echo $(date) ' ' $1
}
# Check if elasticsearch is running before proceeding
nc -zc ${elasticsearch_host} ${elasticsearch_port} &> /dev/null
if [ $? -ne 0 ]
then
    log 'elasticsearch is not running, or it is not running on ${elastic_address}.'
    exit 1
fi

log 'Extracting features...'

# Create training data by using elasticsearch to generate features for RankLib
python3 ./ltrfeatures.py --mapping ./pmid-mapping.json \
                         --queries ${queries_json_file} \
                         --elastic-url http://${elasticsearch_address} \
                         --elastic-index ${elasticsearch_index} > training.txt

log 'Training a model......'

# Train a model
# https://sourceforge.net/p/lemur/wiki/RankLib%20How%20to%20use/
java -jar ${ranklib_path} -train training.txt -ranker 6 -save ${model_name}.txt -gmax 1 -metric2t MAP

log 'Uploading model and searching...'

# Now we can upload the model to elasticsearch
python3 ./uploadscript.py --input ${model_name}.txt --elastic-url http://${elasticsearch_address} -v

python3 ./search.py \
        -q ${queries_json_file} \
        --baseline-output ${trec_baseline} \
        --ltr-output ${trec_ltr} \
        --elastic-url http://${elasticsearch_address} \
        --elastic-index ${elasticsearch_index} \
        --training training.txt

log 'Evaluating...'

# the "baseline" is now done
trec_eval -q qrels.txt ${trec_baseline} > ${output_dest}baseline-${timestamp}.txt

# now, evaluate the fallback PICO
trec_eval -q qrels.txt ${trec_ltr} > ${output_dest}ltr-${timestamp}.txt

log 'Done!'