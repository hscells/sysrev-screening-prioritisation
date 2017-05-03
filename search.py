"""
Produce trec_eval result files to compare learning to rank models against the baseline.

Harry Scells
Apr 2017
"""
import argparse
import io
import json
import sys

from collections import namedtuple
from elasticsearch import Elasticsearch
from typing import List

from ltrfeatures import RankLibRow

TrecResult = namedtuple('TrecResult',
                        ['query_id', 'q0', 'document_id', 'rank', 'score', 'label'])


def search_baseline(query: dict, elastic_url: Elasticsearch,
                    index: str) -> List[TrecResult]:
    """
    Simulate the query as it would normally be issued to PubMed.
    :param query: 
    :param elastic_url: 
    :param index: 
    :return: 
    """
    es = Elasticsearch([elastic_url])
    res = es.search(index=index, doc_type='doc', body={'query': query['query']},
                    size=10000, request_timeout=100)

    for rank, hit in enumerate(res['hits']['hits']):
        yield TrecResult(query['query_id'], '0', hit['_id'], rank + 1,
                         hit['_score'], 'baseline')


def search_ltr(query: dict, elastic_url: str, index: str, model: str,
               training: List[RankLibRow], fv_size: int) -> List[TrecResult]:
    """
    Re-rank the result list using an ltr model.
    :param query: 
    :param elastic_url: 
    :param index: 
    :param model: 
    :param training: 
    :param fv_size: 
    :return: 
    """
    es = Elasticsearch([elastic_url])
    features = []

    normalised_features = {}
    for k in range(fv_size):
        normalised_features[str(k)] = 0.0
    pmids = set()

    for row in training:
        pmids.add(row.info)
        for k, v in row.features.items():
            normalised_features[k] += float(v)

    for sum_weights in normalised_features.values():
        features.append({
            'constant_score': {
                'boost': sum_weights / len(pmids),
                'filter': {
                    'terms': {
                        '_id': list(pmids)
                    }
                }
            }
        })

    rescore_query = \
        {
            'query': query['query'],
            'rescore': {
                'query': {
                    'rescore_query': {
                        'ltr': {
                            'model': {
                                'stored': model
                            },
                            'features': features
                        }
                    }
                }
            }
        }

    # pprint(rescore_query)
    res = es.search(index=index, doc_type='doc',
                    size=10000, request_timeout=1000000,
                    body=rescore_query, _source=False, explain=True)

    for rank, hit in enumerate(res['hits']['hits']):
        if hit['_score'] is not None:
            yield TrecResult(query['query_id'], '0', hit['_id'], rank + 1,
                             hit['_score'], 'ltr')


def format_trec_results(results: List[TrecResult]):
    """
    Format python objects to a string
    :param results: A list of TrecResult objects
    :return: A pretty-printed string ready to be written to file
    """
    return '\n'.join(
        ['{}\t{}\t{}\t{}\t{}\t{}'.format(
            r.query_id, r.q0, r.document_id, r.rank, r.score, r.label) for r in results]) + '\n'


def load_training_data(file: io.TextIOWrapper) -> List[RankLibRow]:
    """
    Create a list of ranklib objects where the feature vector is a sparse vector. It is up to
    the user to grok this sparse vector.
    :return: 
    """

    def marshall_ranklib(row: str) -> RankLibRow:
        target, qid, *rest = row.split()
        # extract the info, if one exists
        info = ''
        if rest[-1][0] == '#':
            info = rest[-1]
            info = info.replace('#', '').strip()
        # remove info item
        if info != '':
            rest = rest[:-2]
        # parse the features
        features = {}
        for pair in rest:
            feature_id, value = pair.split(':')
            v = float(value)
            if v > 0:
                features[feature_id] = v
        return RankLibRow(target=target, qid=qid, features=features, info=info)

    # marshall the data into rank lib row objects
    for line in file.readlines():
        ranklib = marshall_ranklib(line)
        yield ranklib


def calc_feature_vector_size(file: io.TextIOWrapper) -> int:
    """
    Calculate the feature vector size by reading the training data of RankLib
    :param file: RankLib training data
    :return: 
    """
    lines = file.readlines()
    row = lines[0].split()
    return len(row[2:-1])


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('-q', '--queries', help='Input queries file from medline2elastic.',
                           default=sys.stdin, type=argparse.FileType('r'))
    argparser.add_argument('--baseline-output', help='Output path for the trec_eval file.',
                           required=True, type=str)
    argparser.add_argument('--ltr-output', help='Output path for the trec_eval file.',
                           required=True, type=str)
    argparser.add_argument('--training', help='Input training data.', type=str,
                           default='training.txt')
    argparser.add_argument('--model', help='The stored model to use for ltr.', type=str,
                           default='model')
    argparser.add_argument('--elastic-url', help='The full url elasticsearch is running on.',
                           type=str, default='http://localhost:9200')
    argparser.add_argument('--elastic-index', help='The index to search in.',
                           type=str, default='med')

    args = argparser.parse_args()

    Q = json.load(args.queries)

    with open(args.baseline_output, 'w') as f:
        for q in Q:
            f.write(
                format_trec_results(
                    search_baseline(q, args.elastic_url, args.elastic_index)))

    with open(args.training, 'r') as t:
        T = list(load_training_data(t))
    with open(args.training, 'r') as t:
        S = calc_feature_vector_size(t)
    with open(args.ltr_output, 'w') as f:
        for q in Q:
            f.write(
                format_trec_results(
                    search_ltr(q, args.elastic_url, args.elastic_index, args.model,
                               T, S)))
