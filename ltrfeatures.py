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
import importlib
import inspect
import json
import pkgutil
import sys
from functools import partial
from urllib.parse import urljoin

import progressbar
import requests
from collections import namedtuple, OrderedDict
from elasticsearch import Elasticsearch
from typing import List, Dict

from .features.feature import AbstractFeature

RankLibRow = namedtuple('RankLibRow', ['target', 'qid', 'features', 'info'])


def generate_query_vocabulary(queries: List[OrderedDict]) -> OrderedDict:
    """
    Given a query, extract the vocabulary (query terms). The result of this function is an 
    OrderedDict where the keys are the fields in the query and the values are the query terms
    inside those fields being searched on.
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
                    # this hack of a line just visits the node at the next depth, it's like this
                    # because of the way elasticsearch represents match and match_phrase queries
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

    return OrderedDict(sorted(vocabulary.items()), key=lambda x: x[0])


def feature_vector_mapping(mapping: OrderedDict, features: Dict[str, AbstractFeature],
                           queries: List[OrderedDict]) -> OrderedDict:
    """
    Given a query vocabulary (the terms and phrases in a boolean query) and some features,
    and depending on the type of feature, create a mapping to the features in the RankLib
    training data.
    
    Note that the pmid variable stands for PubMed Identifier - for general purposes this can
    be considered the document id.
    :return: 
    """
    inverted_vocabulary = OrderedDict()
    idx = 1
    for query in queries:
        vocab = generate_query_vocabulary([OrderedDict([('query', query['query'])])])
        query_fields = set()
        for k in sorted(vocab.keys()):
            query_fields.add(k)

        for feature_name in sorted(features):
            if feature_name not in inverted_vocabulary:
                inverted_vocabulary[feature_name] = idx
                idx += 1
    return inverted_vocabulary


def generate_features(query: OrderedDict, mapping: OrderedDict, fv_mapping: dict,
                      elastic_url: str, elastic_index: str, elastic_doc: str,
                      feature_classes: Dict[str, AbstractFeature]) -> RankLibRow:
    """
    Generate features using elasticsearch. This function uses the features that have been loaded
    and runs the calc() method on each of the classes. It writes one line of a RankLib training
    file to the disk when it is done. This function is thread safe, it should be run in 
    parallel. Note, that for this to happen, you must have an `output_pointer` variable defined.
        
    Note that the pmid variable stands for PubMed Identifier - for general purposes this can
    be considered the document id.
    :return: 
    """
    # create the elasticsearch object
    es = Elasticsearch([elastic_url])

    document_id = str(query['document_id'])
    query_id = query['query_id']
    es_query = query['query']
    judged_documents = mapping[document_id]
    query_terms = generate_query_vocabulary([OrderedDict([('query', es_query)])])
    fields = list(query_terms.keys())

    del query_terms['key']

    for i in range(len(fields)):
        field = fields[i]
        if field in ['population', 'intervention', 'outcomes']:
            fields.append(field + '.stemmed')

    for k in query_terms.keys():
        t = ' '.join(query_terms[k])
        resp = requests.get(urljoin(elastic_url, elastic_index) + '/_analyze',
                            data=json.dumps({
                                'text': t,
                                'analyzer': 'medline_analyser'
                            }))
        query_terms[k] = list(set([x['token'] for x in json.loads(resp.text)['tokens']]))

    # We look the the studies relevant to this document (systematic review)
    for pmid, relevance in judged_documents.items():
        # The elasticsearch term vector api is used to get collection statistics
        statistics = es.termvectors(index=elastic_index, doc_type=elastic_doc, id=pmid,
                                    fields=fields, request_timeout=10000,
                                    body={
                                        'offsets': True,
                                        'payloads': True,
                                        'positions': True,
                                        'term_statistics': True,
                                        'field_statistics': True
                                    })
        if statistics['found']:
            # We create an ordered dictionary to store the features
            # RankLib likes it when the features are ordered by feature id
            features = OrderedDict()

            # now we can go ahead and fill in the feature vector with values
            for feature_name, feature_class in feature_classes.items():
                # Use some dubious python code to dynamically run features
                # noinspection PyCallingNonCallable
                f = feature_class(statistics=statistics, query=es_query,
                                  query_vocabulary=query_terms).calc()

                features[fv_mapping[feature_name]] = f

            relevance = judged_documents[pmid]
            yield RankLibRow(target=relevance, qid=query_id, info=pmid,
                             features=features)


def format_ranklib_row(row: RankLibRow) -> str:
    """
    This function formats a single row to the exact format needed for RankLib training data.
    :param row: 
    :return: 
    """
    features = ''.join(
        ['{}:{} '.format(feature, value) for (feature, value) in row.features.items()])
    return '{} qid:{} {}#{}\n'.format(row.target, row.qid, features, row.info)


def format_ranklib(rows: List[RankLibRow]) -> str:
    """
    Format several RankLib rows at once.
    :param rows: 
    :return: 
    """
    return ''.join([format_ranklib_row(row) for row in rows])


def load_features() -> OrderedDict:
    """
    Load the features from the `features` module at run time. This allows us to load features
    in a general way and separate from this code. In this way, we abstract the feature 
    engineering process from the code that execute the features.
    :return: 
    """
    # these hard-coded values should never change unless you need to rename the package, module,
    # or abstract class names.
    package_path = 'features'
    abstract_module = 'feature'
    abstract_classes = ['AbstractFeature']

    # get a list of the modules in the feature package, except the abstract feature class
    feature_modules = [name for _, name, _ in pkgutil.iter_modules([package_path])]
    feature_modules.remove(abstract_module)

    # import the modules and keep a reference to them
    modules = [importlib.import_module('{}.{}'.format(package_path, f)) for f in
               feature_modules]

    # now we can go and find the classes in the modules
    classes = set()
    [classes.add((x, None)) for x in abstract_classes]
    for feature_module in modules:
        for member in inspect.getmembers(feature_module, inspect.isclass):
            if issubclass(member[1], AbstractFeature):
                classes.add(member)

    # now remove the abstract class from the list of classes so there are no errors
    classes = dict(classes)
    [classes.pop(abstract_class) for abstract_class in abstract_classes]

    ordered_classes = OrderedDict()
    class_names = sorted(classes.keys())
    for n in class_names:
        ordered_classes[n] = classes[n]
    return ordered_classes


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
    argparser.add_argument('--feature-fields', nargs='+',
                           default=['title', 'text', 'population', 'intervention', 'outcomes'],
                           help='Names of elasticsearch fields to use in features')
    args = argparser.parse_args()

    M = json.load(args.mapping, object_pairs_hook=OrderedDict)
    Q = json.load(args.queries, object_pairs_hook=OrderedDict)
    FV = feature_vector_mapping(M, load_features(), Q)
    generate_features_partial = partial(generate_features,
                                        mapping=M,
                                        fv_mapping=FV,
                                        elastic_url=args.elastic_url,
                                        elastic_index=args.elastic_index,
                                        elastic_doc=args.elastic_doc,
                                        feature_classes=load_features())

    extracted_features = []
    bar = progressbar.ProgressBar()
    for q in bar(Q):
        args.output.write(
            format_ranklib(
                generate_features_partial(q)))
