"""
Produce trec_eval result files to compare learning to rank models against the baseline.

Harry Scells
Apr 2017
"""
import argparse
import io
import json
import sys
from functools import partial

from collections import namedtuple
from elasticsearch import Elasticsearch
from multiprocessing import Pool, Lock
from typing import List

from ltrfeatures import RankLibRow

TrecResult = namedtuple('TrecResult',
                        ['query_id', 'q0', 'document_id', 'rank', 'score', 'label'])
OUTPUT_POINTER = io.IOBase
OUTPUT_LOCK = Lock()
TRAINING_POINTER = io.IOBase
TRAINING_LOCK = Lock()


def search_baseline(query: dict, elastic_url: Elasticsearch,
                    index: str) -> None:
    """
    Simulate the query as it would normally be issued to PubMed.
    :param query: 
    :param elastic_url: 
    :param index: 
    :return: 
    """
    es = Elasticsearch([elastic_url])
    results = []
    res = es.search(index=index, doc_type='doc', body={'query': query['query']},
                    size=10000, request_timeout=100)

    for rank, hit in enumerate(res['hits']['hits']):
        results.append(TrecResult(query['query_id'], '0', hit['_id'], rank + 1,
                                  hit['_score'], 'baseline'))
    OUTPUT_LOCK.acquire()
    try:
        OUTPUT_POINTER.write(format_trec_results(results))
    finally:
        OUTPUT_LOCK.release()


def search_ltr(query: dict, elastic_url: str,
               index: str, model: str, training: str) -> None:
    """
    Re-rank the result list using an ltr model.
    :param query: 
    :param elastic_url: 
    :param index: 
    :param model: 
    :param training: 
    :return: 
    """
    es = Elasticsearch([elastic_url])
    results = []
    features = []

    print('getting training data for {}'.format(query['query_id']))
    TRAINING_LOCK.acquire()
    try:
        with open(training, 'r') as f:
            training = load_training_data(f, query['query_id'])
    finally:
        TRAINING_LOCK.release()

    print('got training data for {}'.format(query['query_id']))

    normalised_features = {}
    pmids = set()

    for row in training:
        pmids.add(row.info)
        # print(row.features)
        for k, v in row.features.items():
            if k not in normalised_features:
                normalised_features[k] = 0.0
            normalised_features[k] += float(v)

    for sum_weights in normalised_features.values():
        features.append({
            'constant_score': {
                'boost': sum_weights / len(training),
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
                    body=rescore_query)

    for rank, hit in enumerate(res['hits']['hits']):
        results.append(TrecResult(query['query_id'], '0', hit['_id'], rank + 1,
                                  hit['_score'], 'ltr'))

    print('{} results for query {}'.format(len(results), query['query_id']))
    OUTPUT_LOCK.acquire()
    try:
        OUTPUT_POINTER.write(format_trec_results(results))
    finally:
        OUTPUT_LOCK.release()
    # OUTPUT_POINTER.write(format_trec_results(results))


def format_trec_results(results: List[TrecResult]):
    """
    Format python objects to a string
    :param results: A list of TrecResult objects
    :return: A pretty-printed string ready to be written to file
    """
    return '\n'.join(
        ['{}\t{}\t{}\t{}\t{}\t{}'.format(
            r.query_id, r.q0, r.document_id, r.rank, r.score, r.label) for r in results]) + '\n'


def load_training_data(file: io.TextIOWrapper, query_id: str) -> List[RankLibRow]:
    """
    :return: 
    """

    def marshall_ranklib(row: str) -> RankLibRow:
        target, _, *rest = row.split()
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
            features[feature_id] = float(value)
        return RankLibRow(target=target, qid=query_id, features=features, info=info)

    # marshall the data into rank lib row objects
    training = []
    for line in file.readlines():
        if line.split()[1].split(':')[-1] == str(query_id):
            ranklib = marshall_ranklib(line)
            training.append(ranklib)

    return training


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('-q', '--queries', help='Input queries file from medline2elastic.',
                           default=sys.stdin, type=argparse.FileType('r'))
    argparser.add_argument('--baseline-output', help='Output path for the trec_eval file.',
                           required=True, type=argparse.FileType('w'))
    argparser.add_argument('--ltr-output', help='Output path for the trec_eval file.',
                           required=True, type=argparse.FileType('w'))
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

    TRAINING_POINTER = args.training

    OUTPUT_POINTER = args.baseline_output
    p = Pool()
    baseline_partial = partial(search_baseline,
                               elastic_url=args.elastic_url,
                               index=args.elastic_index)
    p.map(baseline_partial, Q)
    p.close()
    p.join()

    OUTPUT_POINTER = args.ltr_output
    p = Pool()
    ltr_partial = partial(search_ltr,
                          elastic_url=args.elastic_url,
                          index=args.elastic_index,
                          model=args.model,
                          training=args.training)
    p.map(ltr_partial, Q)
    # for q in Q:
    #     ltr_partial(q)
    p.close()
    p.join()
