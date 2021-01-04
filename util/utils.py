import os
import sys
import argparse

try:
    import ailia
    AILIA_EXIST = True
except ImportError:
    # TODO: create logger.py --> @sngyo
    print('[WARNING]: ailia package cannot be found under `sys.path`')
    print('[WARNING]: default env_id is set to 0, you can change the id by '
          '[--env_id N]')
    AILIA_EXIST = False


def check_file_existance(filename):
    if os.path.isfile(filename):
        return True
    else:
        print(f'[ERROR] {filename} not found')
        sys.exit()


def get_base_parser(description, default_input, default_save, parse=False):
    """
    Get ailia default argument parser

    Parameters
    ----------
    description : str
    default_input : str
        default input data (image / video) path
    default_save : str
        default save path
    parse : bool, default is False
        if True, return parsed arguments
        TODO: deprecates

    Returns
    -------
    out : ArgumentParser()

    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=description,
        conflict_handler='resolve',  # allow to overwrite default argument
    )
    parser.add_argument(
        '-i', '--input', metavar='IMAGE/VIDEO', default=default_input,
        help='The default (model-dependent) input data (image / video) path.'
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
    parser.add_argument(
        '-e', '--env_id', type=int,
        default=ailia.get_gpu_environment_id() if AILIA_EXIST else 0,
        help=('A specific environment id can be specified. By default, '
              'the return value of ailia.get_gpu_environment_id will be used')
    )

    if parse:
        parser = parser.parse_args()

    return parser


def update_parser(parser):
    """Default check or update configurations should be placed here

    Parameters
    ----------
    parser : ArgumentParser()

    Returns
    -------
    args : ArgumentParser()
        (parse_args() will be done here)
    """
    args = parser.parse_args()

    # 1. check env_id count
    if AILIA_EXIST:
        count = ailia.get_environment_count()
        if count <= args.env_id:
            print(f'[ERROR] specified env_id: {args.env_id} cannot found. ')
            print('env_id updated to 0')

    print(f'env_id: {args.env_id}')

    return args
