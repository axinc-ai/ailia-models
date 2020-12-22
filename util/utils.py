import os
import sys
import argparse
import glob
from logging import DEBUG

from params import MODALITIES, EXTENSIONS
import log_init

# FIXME: Next two lines should be better to call from the main script
# once we prepared one. For now, we do the initialization of logger here.
logger = log_init.logger
logger.info('Start!')

# TODO: better to use them (first, fix above)
# from logging import getLogger
# logger = getLogger(__name__)


# TODO: yaml config file and yaml loader

try:
    import ailia
    AILIA_EXIST = True
except ImportError:
    logger.warning('ailia package cannot be found under `sys.path`')
    logger.warning('default env_id is set to 0, you can change the id by '
                   '[--env_id N]')
    AILIA_EXIST = False


def check_file_existance(filename):
    if os.path.isfile(filename):
        return True
    else:
        logger.error(f'{filename} not found')
        sys.exit()


def get_base_parser(
        description, default_input, default_save, input_ftype='image',
):
    """
    Get ailia default argument parser

    Parameters
    ----------
    description : str
    default_input : str
        default input data (image / video) path
    default_save : str
        default save path
    input_ftype : str

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
        help=('The default (model-dependent) input data (image / video) path. '
              'If a directory name is specified, the model will be run for '
              'the files inside. File type is specified by --ftype argument')
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
    parser.add_argument(
        '--ftype', metavar='FILE_TYPE', default=input_ftype,
        choices=MODALITIES,
        help='file type list: ' + ' | '.join(MODALITIES)
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='set default logger level to DEBUG (enable to show DEBUG logs)'
    )
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

    # -------------------------------------------------------------------------
    # 0. logger level update
    if args.debug:
        logger.setLevel(DEBUG)

    # -------------------------------------------------------------------------
    # 1. check env_id count
    if AILIA_EXIST:
        count = ailia.get_environment_count()
        if count <= args.env_id:
            logger.error(f'specified env_id: {args.env_id} cannot found. ')
            logger.info('env_id updated to 0')
            args.env_id = 0

    logger.info(f'env_id: {args.env_id}')

    # -------------------------------------------------------------------------
    # 2. update input
    if isinstance(args.input, list):
        # LIST --> nothing will be changed here.
        pass
    elif os.path.isdir(args.input):
        # Directory Path --> generate list of inputs
        files_grapped = []
        for extension in EXTENSIONS[args.ftype]:
            files_grapped.extend(
                glob.glob(os.path.join(args.input, extension))
            )
        logger.info(f'{len(files_grapped)} {args.ftype} files found!')

        args.input = files_grapped

        # create save directory
        if args.savepath is None:
            pass
        else:
            if '.' in args.savepath:
                logger.warning('Please specify save directory as --savepath '
                               'if you specified a direcotry for --input')
                logger.info('[./results] directory will be created')
                args.savepath = 'results'
            os.makedirs(args.savepath, exist_ok=True)
            logger.info(f'output saving directory: {args.savepath}')

    elif os.path.isfile(args.input):
        args.input = [args.input]
    else:
        logger.error('specified input is not file path nor directory path')

    # -------------------------------------------------------------------------
    return args
