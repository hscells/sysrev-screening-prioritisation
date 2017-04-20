"""
Create training data for RankLib

------------------------------------------------------------------------------------------------
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

------------------------------------------------------------------------------------------------

Harry Scells
Apr 2017
"""

import argparse
import json
import sys
from copy import deepcopy
from functools import partial

from collections import namedtuple, OrderedDict
from elasticsearch import Elasticsearch
from multiprocessing import Pool
from typing import List, Dict

RankLibRow = namedtuple('RankLibRow', ['target', 'qid', 'features', 'info'])
FeatureIdentifier = namedtuple('FeatureIdentifier', ['weight', 'field', 'term'])

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

output_pointer = object


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


def generate_query_vocabulary(queries: List[OrderedDict]) -> OrderedDict:
    """
    
    :param queries: 
    :return: 
    """

    # we store the vocabulary outside of the tree traversal for convenience
    vocabulary = OrderedDict()

    def recurse(node: Dict) -> None:
        """
        Traverse the query tree, adding terms and the related fields to the vocabulary as we go.
        :param node: Pointer to the node in the tree to continue to recurse down.
        """
        # keep diving if it's a list
        if type(node) is list:
            for item in node:
                recurse(item)
        # otherwise it might be a match
        elif type(node) is dict or type(node) is OrderedDict:
            # when we find a terminal leaf, add it to the vocabulary
            for key in node.keys():
                if key == 'match' or key == 'match_phrase':
                    terminals = list(node[list(node.keys())[0]].items())[0]
                    field, term = terminals[0].replace('.stemmed', ''), terminals[1]
                    if field not in vocabulary:
                        vocabulary[field] = set()
                    vocabulary[field].add(term)
                elif key == 'multi_match':
                    fields, term = node[key]['fields'], node[key]['query']
                    for field in fields:
                        field = field.replace('.stemmed', '')
                        if field not in vocabulary:
                            vocabulary[field] = set()
                        vocabulary[field].add(term)
                # if the node wasn't terminal, dive deeper
                else:
                    recurse(node[key])
        else:
            return

    # traverse down the query tree to find the terms and matching fields
    for query in queries:
        recurse(query['query'])

    for k, v in vocabulary.items():
        vocabulary[k] = sorted(v)

    return vocabulary


def feature_identifier(weight: str, field: str, term: str) -> str:
    return '{}{}{}'.format(weight, field.upper().replace('.', ''), ''.join(term))


def map_query_vocabulary_to_features(vocabulary: OrderedDict, weights: list,
                                     start_index=2) -> OrderedDict:
    """
    
    :param vocabulary: 
    :param weights: 
    :param start_index: 
    :return: 
    """
    inverted_vocabulary = OrderedDict()
    for field, terms in vocabulary.items():
        for term in terms:
            for weight in weights:
                inverted_vocabulary[feature_identifier(weight, field, term)] = start_index
                start_index += 1
    return inverted_vocabulary


def map_identifier_to_feature(weight: str, field: str, term: str,
                              inverted_vocabulary: dict) -> int:
    """
    
    :param weight: 
    :param field: 
    :param term: 
    :param inverted_vocabulary: 
    :return: 
    """
    return inverted_vocabulary[feature_identifier(weight, field, term)]


def generate_features(query: OrderedDict, mapping: OrderedDict, fv_mapping: OrderedDict,
                      elastic_url: str, elastic_index: str) -> None:
    """
    From the data exported from medline2elastic, extract the doc ids
    :return: 
    """
    # create the elasticsearch object
    es = Elasticsearch([elastic_url])

    # the names of the weights we will use

    document_id = str(query['document_id'])
    query_id = query['query_id']
    es_query = query['query']
    judged_documents = mapping[document_id]
    judged_document_ids = list(judged_documents.keys())

    # get a rank score (for a feature)
    res = es.search(index=elastic_index,
                    body=populate_feature_query(judged_document_ids, es_query))

    for pmid, relevance in judged_documents.items():
        features = OrderedDict()
        for i in range(len(fv_mapping)):
            features[i] = 0
        # score
        features[1] = 0
        # hits
        features[2] = res['hits']['total']

        # populate the score feature
        for retrieved_document in res['hits']['hits']:
            if retrieved_document['_id'] == pmid:
                features[1] = retrieved_document['_score']

        # get elasticsearch statistics
        statistics = es.termvectors(index=elastic_index, doc_type='doc', id=pmid)

        # we need a subset of the vocabulary to work on
        query_terms = generate_query_vocabulary([OrderedDict([('query', es_query)])])
        for field, terms in query_terms.items():
            for term in terms:
                if statistics['found']:
                    tv = statistics['term_vectors']
                    df = map_identifier_to_feature('df', field, term,
                                                   fv_mapping)
                    tf = map_identifier_to_feature('tf', field, term,
                                                   fv_mapping)
                    # idf = map_identifier_to_feature('idf', field, term,
                    #                                 feature_vocabulary_mapping)
                    features[df] = tv[field]['field_statistics']['sum_doc_freq']
                    features[tf] = tv['terms'][term]['term_freq']

        output_pointer.write(
            format_ranklib_row(RankLibRow(target=relevance, qid=query_id, features=features,
                                          info=pmid)))


def format_ranklib_row(row: RankLibRow) -> str:
    """
    
    :param row: 
    :return: 
    """
    features = ''.join(
        ['{}:{} '.format(feature, value) for (feature, value) in row.features.items()])
    return '{} qid:{} {}# {}\n'.format(row.target, row.qid, features, row.info)


def format_ranklib(rows: List[RankLibRow]) -> str:
    """
    
    :param rows: 
    :return: 
    """
    return ''.join([format_ranklib_row(row) for row in rows])


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Create training data for RankLib')

    # create this from `extract_pmids.py` in sysrev-collections
    argparser.add_argument('-m', '--mapping', help='The mapping file', required=True,
                           type=argparse.FileType('r'))
    # download this from medline2elastic at /api/queries/elastic/pico
    argparser.add_argument('-q', '--queries', help='The queries file', required=True,
                           type=argparse.FileType('r'))
    argparser.add_argument('-o', '--output', help='The file to output to', default=sys.stdout,
                           type=argparse.FileType('w+'))
    argparser.add_argument('--elastic-url', help='Address of the elasticsearch instance',
                           default='http://localhost:9200', type=str)
    argparser.add_argument('--elastic-index', help='Index to train using',
                           default='med', type=str)

    args = argparser.parse_args()

    output_pointer = args.output

    input_queries = json.load(args.queries, object_pairs_hook=OrderedDict)
    # generate a vocabulary from the queries and a mapping to features for RankLib
    query_vocabulary = generate_query_vocabulary(input_queries)
    feature_vocabulary_mapping = map_query_vocabulary_to_features(vocabulary=query_vocabulary,
                                                                  weights=['df', 'tf', 'idf'])

    generate_features_partial = partial(generate_features,
                                        mapping=json.load(args.mapping,
                                                          object_pairs_hook=OrderedDict),
                                        fv_mapping=feature_vocabulary_mapping,
                                        elastic_url=args.elastic_url,
                                        elastic_index=args.elastic_index)

    p = Pool()
    annotated_data = p.map(generate_features_partial, input_queries)
    p.close()
    p.join()
