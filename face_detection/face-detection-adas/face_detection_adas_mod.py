import sys
from importlib import import_module

import utils


def dummy_parser(parser, *args):
    args = parser.parse_args()
    return args


argv = sys.argv
update_parser = utils.update_parser
try:
    sys.argv = sys.argv[:1]
    utils.update_parser = dummy_parser

    mod = import_module("face-detection-adas")
finally:
    sys.argv = argv
    utils.update_parser = update_parser
