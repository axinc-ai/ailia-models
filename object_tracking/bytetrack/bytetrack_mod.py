import sys
from importlib import import_module

import arg_utils

mod = None


def set_args(args):
    mod.args = args


def dummy_parser(parser, *args):
    args = parser.parse_args()
    return args


argv = sys.argv
update_parser = arg_utils.update_parser
try:
    sys.argv = sys.argv[:1]
    arg_utils.update_parser = dummy_parser

    mod = import_module("bytetrack")
finally:
    sys.argv = argv
    arg_utils.update_parser = update_parser
