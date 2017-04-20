"""
Create training data for RankLib

------------------------------------------------------------------------------------------------

The file format for the training data (also testing/validation data) is the same as for 
SVM-Rank. This is also the format used in LETOR datasets. Each of the following lines represents
one training example and is of the following format:

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

------------------------------------------------------------------------------------------------

Harry Scells
Apr 2017
"""

import argparse
import json
import sys
from copy import deepcopy

from collections import namedtuple, OrderedDict
from elasticsearch import Elasticsearch
from typing import List

RankLibRow = namedtuple('RankLibRow', ['target', 'qid', 'features', 'info'])

feature_query = {
    'query': {
        'bool': {
            'filter': {
                'ids': {
                    'values': [
                        # list of ids goes here
                    ]
                }
            },
            'should': {
                # query goes here
            }
        }
    }
}


def populate_feature_query(document_ids: List[str], query: dict) -> dict:
    """
    
    :param document_ids: 
    :param query: 
    :return: 
    """
    base = deepcopy(feature_query)
    base['query']['bool']['filter']['ids']['values'] = document_ids
    base['query']['bool']['should'] = query
    return base


def generate_features(es_url: str, idx: str, mapping: dict, queries: dict) -> List[RankLibRow]:
    """
    From the data exported from medline2elastic, extract the doc ids
    :return: 
    """
    es = Elasticsearch([es_url])
    ranklib_training = []
    for query in queries:
        document_id = str(query['document_id'])
        query_id = query['query_id']
        es_query = query['query']
        judged_documents = mapping[document_id]
        judged_document_ids = list(judged_documents.keys())

        es_query['bool'].pop('must_not')

        res = es.search(index=idx,
                        body=populate_feature_query(judged_document_ids, es_query))

        for pmid, relevance in judged_documents.items():
            features = OrderedDict()
            # score
            features[1] = 0
            # hits
            features[2] = res['hits']['total']
            for retrieved_document in res['hits']['hits']:
                if retrieved_document['_id'] == pmid:
                    features[1] = retrieved_document['_score']

            ranklib_training.append(
                RankLibRow(target=relevance, qid=query_id, features=features, info=pmid))

    return ranklib_training


def format_ranklib_row(row: RankLibRow) -> str:
    """
    
    :param row: 
    :return: 
    """
    features = ''.join(
        ['{}:{} '.format(feature, value) for (feature, value) in row.features.items()])
    return '{} qid:{} {}# {}'.format(row.target, row.qid, features, row.info)


def format_ranklib(rows: List[RankLibRow]) -> str:
    """
    
    :param rows: 
    :return: 
    """
    return '\n'.join([format_ranklib_row(row) for row in rows])


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Create training data for RankLib')

    # create this from `extract_pmids.py` in sysrev-collections
    argparser.add_argument('-m', '--mapping', help='The mapping file', required=True,
                           type=argparse.FileType('r'))
    # download this from medline2elastic at /api/queries/elastic/pico
    argparser.add_argument('-q', '--queries', help='The queries file', required=True,
                           type=argparse.FileType('r'))
    argparser.add_argument('-o', '--output', help='The file to output to', default=sys.stdout,
                           type=argparse.FileType('w'))
    argparser.add_argument('--elastic-url', help='Address of the elasticsearch instance',
                           default='http://localhost:9200', type=str)
    argparser.add_argument('--elastic-index', help='Index to train using',
                           default='med', type=str)

    args = argparser.parse_args()

    args.output.write(
        format_ranklib(
            generate_features(
                args.elastic_url,
                args.elastic_index,
                json.load(args.mapping),
                json.load(args.queries))))
