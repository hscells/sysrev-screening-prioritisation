"""
Create training data for RankLib

https://sourceforge.net/p/lemur/wiki/RankLib%20File%20Format/

The file format for the training data (also testing/validation data) is the same as for 
SVM-Rank. This is also the format used in LETOR data sets. Each of the following lines 
represents one training example and is of the following format:

<line> .=. <target> qid:<qid> <feature>:<value> <feature>:<value> ... <feature>:<value> # <info>
<target> .=. <positive integer>
<qid> .=. <positive integer>
<feature> .=. <positive integer>
<value> .=. <float>
<info> .=. <string>

Here's an example: (taken from the SVM-Rank website). Note that everything after "#" are
ignored.

3 qid:1 1:1 2:1 3:0 4:0.2 5:0 # 1A
2 qid:1 1:0 2:0 3:1 4:0.1 5:1 # 1B 
1 qid:1 1:0 2:1 3:0 4:0.4 5:0 # 1C
1 qid:1 1:0 2:0 3:1 4:0.3 5:0 # 1D  
1 qid:2 1:0 2:0 3:1 4:0.2 5:0 # 2A  
2 qid:2 1:1 2:0 3:1 4:0.4 5:0 # 2B 
1 qid:2 1:0 2:0 3:1 4:0.1 5:0 # 2C 
1 qid:2 1:0 2:0 3:1 4:0.2 5:0 # 2D  
2 qid:3 1:0 2:0 3:1 4:0.1 5:1 # 3A 
3 qid:3 1:1 2:1 3:0 4:0.3 5:0 # 3B 
4 qid:3 1:1 2:0 3:0 4:0.4 5:1 # 3C 
1 qid:3 1:0 2:1 3:1 4:0.5 5:0 # 3D

This program uses feature classes located in the `features` folder. For an explanation of how to
extend the features, read the source code of `./features/feature.py`. These feature classes are 
used to extract features. They are separate from this code to abstract away individual `units`
of features. In this way, we can do composable feature engineering without worrying how the 
features are executed, only the assumptions that the classes provide.

Harry Scells
Apr 2017
"""

import argparse
import json
from functools import partial
from string import Template

import os
import sys
from collections import namedtuple, OrderedDict
from elasticsearch import Elasticsearch
from multiprocessing import Pool
from typing import List, Dict

RankLibRow = namedtuple('RankLibRow', ['target', 'qid', 'features', 'info'])
output_pointer = 0


def template_query(query: dict, template: str) -> dict:
    """
    
    :param query: 
    :param template: 
    :return: 
    """
    return json.loads(Template(template).substitute(query=json.dumps(query)))


def generate_features(query: OrderedDict, mapping: OrderedDict,
                      elastic_url: str, elastic_index: str, elastic_doc: str,
                      feature_classes: Dict[int, str]) -> List[RankLibRow]:
    """

    :return: 
    """
    # create the elasticsearch object
    es = Elasticsearch([elastic_url])

    # the names of the weights we will use

    document_id = str(query['document_id'])
    query_id = query['query_id']
    es_query = query['query']
    judged_documents = mapping[document_id]

    docs = {}

    for feature_id, feature_query in feature_classes.items():

        features = OrderedDict()
        res = es.search(index=elastic_index, doc_type=elastic_doc,
                        body=template_query(es_query, feature_query),
                        size=10000, request_timeout=10000)
        for rank, hit in enumerate(res['hits']['hits']):
            pmid = hit['_id']
            if pmid in judged_documents:
                features[feature_id] = hit['_score']
                docs[pmid] = features

    ranklib = []
    for pmid, features in docs.items():
        relevance = judged_documents[pmid]
        ranklib.append(RankLibRow(target=relevance, qid=query_id, features=features, info=pmid))
    return ranklib


def format_ranklib_row(row: RankLibRow) -> str:
    """
    This function formats a single row to the exact format needed for RankLib training data.
    :param row: 
    :return: 
    """
    features = ''.join(
        ['{}:{} '.format(feature, value) for (feature, value) in row.features.items()])
    return '{} qid:{} {}# {}\n'.format(row.target, row.qid, features, row.info)


def format_ranklib(rows: List[RankLibRow]) -> str:
    """
    Format several RankLib rows at once.
    :param rows: 
    :return: 
    """
    return ''.join([format_ranklib_row(row) for row in rows])


def load_features() -> Dict[int, str]:
    """
    :return: 
    """
    features = {}
    for root, dirs, files in os.walk('features'):
        for file in files:
            path = os.path.join(root, file)
            with open(path) as f:
                features[len(features) + 1] = f.read()
    return features


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Create training data for RankLib')

    # create this from `extract_pmids.py` in sysrev-collections
    argparser.add_argument('-m', '--mapping', help='The mapping file', required=True,
                           type=argparse.FileType('r'))
    # download this from medline2elastic at /api/queries/elastic/pico
    argparser.add_argument('-q', '--queries', help='The queries file', required=True,
                           type=argparse.FileType('r'))
    argparser.add_argument('-o', '--output', help='The file to output to',
                           type=argparse.FileType('w'), default=sys.stdout)
    argparser.add_argument('--elastic-url', help='Address of the elasticsearch instance',
                           default='http://localhost:9200', type=str)
    argparser.add_argument('--elastic-index', help='Index to train using',
                           default='med', type=str)
    argparser.add_argument('--elastic-doc', help='Type of the elasticsearch document',
                           default='doc', type=str)

    args = argparser.parse_args()

    custom_features = load_features()

    input_queries = json.load(args.queries)

    generate_features_partial = partial(generate_features,
                                        mapping=json.load(args.mapping),
                                        elastic_url=args.elastic_url,
                                        elastic_index=args.elastic_index,
                                        elastic_doc=args.elastic_doc,
                                        feature_classes=custom_features)

    p = Pool()
    extracted_features = p.map(generate_features_partial, input_queries)
    p.close()
    p.join()

    args.output.write(
        format_ranklib(
            [item for sublist in extracted_features for item in sublist]))
