import argparse
import sys
from urllib.parse import urljoin

import requests

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('-i', '--input', help='Input model', default=sys.stdin,
                           type=argparse.FileType('r'))
    argparser.add_argument('--elastic-url', type=str, default='http://localhost:9200')
    argparser.add_argument('--model-name', type=str, default='model')

    args = argparser.parse_args()

    data = args.input.read()

    response = requests.put(urljoin(args.elastic_url, '/_scripts/ranklib/', args.model_name),
                            data={'script': data})

    print(response.text)
