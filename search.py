"""
Produce trec_eval result files to compare learning to rank models against the baseline.

Harry Scells
Apr 2017
"""
import argparse
import io
import json
import sys

import progressbar
from collections import namedtuple, OrderedDict
from elasticsearch import Elasticsearch
from typing import List, Dict

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
                    size=1000, request_timeout=10000, _source=False)

    for rank, hit in enumerate(res['hits']['hits']):
        yield TrecResult(query_id=query['document_id'], q0='0', document_id=hit['_id'],
                         rank=rank + 1, score=hit['_score'], label='baseline')


def search_ltr(query: dict, elastic_url: str, index: str, model: str,
               training: Dict[int, RankLibRow], fv_size: int) -> List[TrecResult]:
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
    features = OrderedDict()

    # Build the feature queries, here we are basically simulating the results list using a
    # boolean query. The constant_score is the score being assigned as if there was a ranking
    # function. This is just the way the elasticsearch ltr plugin works.
    for k in range(1, fv_size + 1):
        feature_queries = []
        for row in training[query['query_id']]:
            if k in row.features:
                feature_queries.append({
                    'constant_score': {
                        'boost': row.features[k],
                        'filter': {
                            'match': {
                                '_id': row.info
                            }
                        }
                    }
                })
        features[k] = \
            {
                "bool": {
                    "should": feature_queries
                }
            }

    # Once we have all the individual feature queries, we can build the rescore query.
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
                            'features': list(features.values())
                        }
                    }
                }
            }
        }

    # pprint(rescore_query)
    res = es.search(index=index, doc_type='doc',
                    size=1000, request_timeout=10000000,
                    body=rescore_query, _source=False)

    for rank, hit in enumerate(res['hits']['hits']):
        if hit['_score'] is not None:
            yield TrecResult(query_id=query['document_id'], q0='0', document_id=hit['_id'],
                             rank=rank + 1, score=hit['_score'], label='ltr')


def format_trec_results(results: List[TrecResult]):
    """
    Format python objects to a string
    :param results: A list of TrecResult objects
    :return: A pretty-printed string ready to be written to file
    """
    return ''.join(
        ['{}\t{}\t{}\t{}\t{}\t{}\n'.format(
            r.query_id, r.q0, r.document_id, r.rank, r.score, r.label) for r in results])


def load_training_data(file: io.TextIOWrapper) -> Dict[int, RankLibRow]:
    """
    Create a list of ranklib objects where the feature vector is a sparse vector. It is up to
    the user to grok this sparse vector.
    :return: 
    """

    def marshall_ranklib(row: str) -> RankLibRow:
        """
        This nasty looking function just reads a feature file line and marshalls it into a 
        Ranklib object.
        :param row: 
        :return: 
        """
        target, qid, *rest = row.split()
        qid = int(qid.split(':')[-1])
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
                features[int(feature_id)] = v
        return RankLibRow(target=target, qid=qid, features=features, info=info)

    ranklib_rows = {}

    # marshall the data into rank lib row objects
    bar = progressbar.ProgressBar()
    for line in bar(file.readlines()):
        ranklib = marshall_ranklib(line)
        if ranklib.qid not in ranklib_rows:
            ranklib_rows[ranklib.qid] = []
        ranklib_rows[ranklib.qid].append(ranklib)

    return ranklib_rows


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
    print('Searching Baseline...')
    with open(args.baseline_output, 'w') as f:
        bar = progressbar.ProgressBar()
        for q in bar(Q):
            f.write(
                format_trec_results(
                    search_baseline(q, args.elastic_url, args.elastic_index)))

    print('Reading features...')
    with open(args.training, 'r') as t:
        T = load_training_data(t)
    with open(args.training, 'r') as t:
        S = calc_feature_vector_size(t)

    print('Searching LTR...')
    with open(args.ltr_output, 'w') as f:
        bar = progressbar.ProgressBar()
        for q in bar(Q):
            f.write(
                format_trec_results(
                    search_ltr(q, args.elastic_url, args.elastic_index, args.model,
                               T, S)))
