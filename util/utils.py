import os
import sys
import argparse


def check_file_existance(filename):
    if os.path.isfile(filename):
        return True
    else:
        print(f'[ERROR] {filename} not found')
        sys.exit()


def get_base_parser(description, default_img, default_save, parse=False):
    """
    Get ailia default argument parser

    Parameters
    ----------
    description : str
    parse : bool, default is False
        if True, return parsed arguments

    Returns
    -------
    out : ArgumentParser()

    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=description,
    )
    parser.add_argument(
        '-i', '--input', metavar='IMAGE', default=default_img,
        help='The input image path.'
    )
    parser.add_argument(
        '-v', '--video', metavar='VIDEO', default=None,
        help=('You can convert the input video by entering style image.'
              'If the int variable is given, '
              'corresponding webcam input will be used.')
    )
    parser.add_argument(
        '-s', '--savepath', metavar='SAVE_PATH', default=default_save,
        help='Save path for the output (image / video).'
    )
    parser.add_argument(
        '-b', '--benchmark', action='store_true',
        help=('Running the inference on the same input 5 times to measure '
              'execution performance. (Cannot be used in video mode)')
    )

    if parse:
        parser = parser.parse_args()

    return parser
