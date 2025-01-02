import os
import urllib.request
import ssl
import shutil

# logger
from logging import getLogger
logger = getLogger(__name__)


def progress_print(block_count, block_size, total_size):
    """
    Callback function to display the progress
    (ref: https://qiita.com/jesus_isao/items/ffa63778e7d3952537db)

    Parameters
    ----------
    block_count:
    block_size:
    total_size:
    """
    percentage = 100.0 * block_count * block_size / total_size
    if percentage > 100:
        # Bigger than 100 does not look good, so...
        percentage = 100
    max_bar = 50
    bar_num = int(percentage / (100 / max_bar))
    progress_element = '=' * bar_num
    if bar_num != max_bar:
        progress_element += '>'
    bar_fill = ' '  # fill the blanks
    bar = progress_element.ljust(max_bar, bar_fill)
    total_size_kb = total_size / 1024
    print(f'[{bar} {percentage:.2f}% ( {total_size_kb:.0f}KB )]', end='\r')


def urlretrieve(remote_path, weight_path, progress_print):
    temp_path = weight_path + ".tmp"
    try:
        #raise ssl.SSLError # test
        urllib.request.urlretrieve(
            remote_path,
            temp_path,
            progress_print,
        )
    except ssl.SSLError as e:
        logger.info(f'SSLError detected, so try to download without ssl')
        remote_path = remote_path.replace("https","http")
        urllib.request.urlretrieve(
            remote_path,
            temp_path,
            progress_print,
        )
    shutil.move(temp_path, weight_path)


def check_and_download_models(weight_path, model_path, remote_path):
    """
    Check if the onnx file and prototxt file exists,
    and if necessary, download the files to the given path.

    Parameters
    ----------
    weight_path: string
        The path of onnx file.
    model_path: string
        The path of prototxt file for ailia.
    remote_path: string
        The url where the onnx file and prototxt file are saved.
        ex. "https://storage.googleapis.com/ailia-models/mobilenetv2/"
    """

    if not os.path.exists(weight_path):
        logger.info(f'Downloading onnx file... (save path: {weight_path})')
        urlretrieve(
            remote_path + os.path.basename(weight_path),
            weight_path,
            progress_print,
        )
        logger.info('\n')
    if model_path!=None and not os.path.exists(model_path):
        logger.info(f'Downloading prototxt file... (save path: {model_path})')
        urlretrieve(
            remote_path + os.path.basename(model_path),
            model_path,
            progress_print,
        )
        logger.info('\n')
    logger.info('ONNX file and Prototxt file are prepared!')


def check_and_download_file(file_path, remote_path):
    """
    Check if the file exists,
    and if necessary, download the files to the given path.

    Parameters
    ----------
    file_path: string
        The path of file.
    remote_path: string
        The url where the file is saved.
        ex. "https://storage.googleapis.com/ailia-models/mobilenetv2/"
    """

    if not os.path.exists(file_path):
        logger.info('Downloading %s...' % file_path)
        urlretrieve(remote_path + os.path.basename(file_path), file_path, progress_print)
    logger.info('%s is prepared!' % file_path)
