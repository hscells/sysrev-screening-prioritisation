"""
Produce trec_eval result files to compare learning to rank models against the baseline.

Harry Scells
Apr 2017
"""
import argparse
import json
import sys

from collections import namedtuple
from elasticsearch import Elasticsearch
from typing import List

TrecResult = namedtuple('TrecResult',
                        ['query_id', 'q0', 'document_id', 'rank', 'score', 'label'])


def search_baseline(query: dict, es: Elasticsearch, index: str) -> List[TrecResult]:
    """
    Simulate the query as it would normally be issued to PubMed.
    :param index: 
    :param es: 
    :param query: 
    :return: 
    """
    res = es.search(index=index, doc_type='doc', body={'query': query['query']},
                    size=100, request_timeout=100)

    for rank, hit in enumerate(res['hits']['hits']):
        yield TrecResult(query['document_id'], '0', hit['_id'], rank + 1,
                         hit['_score'], 'baseline')


def search_ltr(query: dict, es: Elasticsearch, index: str, model: str) -> List[TrecResult]:
    """
    Re-rank the result list using an ltr model.
    :param query: 
    :param es: 
    :param index: 
    :param model: 
    :return: 
    """
    res = es.search(index=index, doc_type='doc',
                    size=100, request_timeout=100,
                    body={
                        "query": {
                            "ltr": {
                                "model": {
                                    "stored": model
                                },
                                "features": [{"constant_score": {query['query']}}]
                            }
                        }
                    })

    for rank, hit in enumerate(res['hits']['hits']):
        yield TrecResult(query['document_id'], '0', hit['_id'], rank + 1,
                         hit['_score'], 'baseline')


def format_trec_results(results: List[TrecResult]):
    """
    Format python objects to a string
    :param results: A list of TrecResult objects
    :return: A pretty-printed string ready to be written to file
    """
    return '\n'.join(
        ['{}\t{}\t{}\t{}\t{}\t{}'.format(
            r.query_id, r.q0, r.document_id, r.rank, r.score, r.label) for r in results])


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('-q', '--queries', help='Input queries file from medline2elastic.',
                           default=sys.stdin, type=argparse.FileType('r'))
    argparser.add_argument('--baseline-output', help='Output path for the trec_eval file.',
                           required=True, type=argparse.FileType('w'))
    argparser.add_argument('--ltr-output', help='Output path for the trec_eval file.',
                           required=True, type=argparse.FileType('w'))
    argparser.add_argument('--model', help='The stored model to use for ltr.', type=str,
                           default='model')
    argparser.add_argument('--elastic-url', help='The full url elasticsearch is running on.',
                           type=str, default='http://localhost:9200')
    argparser.add_argument('--elastic-index', help='The index to search in.',
                           type=str, default='med')

    args = argparser.parse_args()

    args.baseline_output.write(
        format_trec_results(
            search_baseline(
                json.load(args.queries),
                Elasticsearch([args.elastic_url]),
                args.elastic_index)))

    args.ltr_output.write(
        format_trec_results(
            search_ltr(
                json.load(args.queries),
                Elasticsearch([args.elastic_url]),
                args.elastic_index,
                args.model)))
