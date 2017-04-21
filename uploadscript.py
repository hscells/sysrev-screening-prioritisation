"""
The RankLib plugin for elasticsearch (or maybe just ranklib itself) is very finicky about the 
exact way that the data for the model needs to be uploaded. This little script handles this 
process for you do there is no stuffing about in bash or kibana >:(

Harry Scells
Apr 2017
"""

import argparse
import json
import sys
from urllib.parse import urljoin

import requests

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('-i', '--input', help='Input model', default=sys.stdin,
                           type=argparse.FileType('r'))
    argparser.add_argument('--elastic-url', help='The full url elasticsearch is running on',
                           type=str, default='http://localhost:9200')
    argparser.add_argument('--model-name', help='The name of the model in elasticsearch',
                           type=str, default='model')
    argparser.add_argument('-v', help='Verbose?', action='store_true')

    args = argparser.parse_args()

    data = args.input.read()
    parsed_data = json.dumps(
        {'script': '\n'.join([line for line in data.splitlines()]).replace('\t', ' ')})

    response = requests.put(urljoin(args.elastic_url,
                                    '{}{}{}'.format('/_scripts/ranklib/', args.model_name,
                                                    '?pretty=true')),
                            data=parsed_data)

    if args.v:
        print(response.text)
