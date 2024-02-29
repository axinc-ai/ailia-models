import subprocess
from logging import getLogger

logger = getLogger(__name__)


def run_ffmpeg(args) -> bool:
    commands = [ 'ffmpeg', '-hide_banner', '-loglevel', 'error' ]
    commands.extend(args)
    try:
        subprocess.run(commands, stderr = subprocess.PIPE, check = True)
        return True
    except subprocess.CalledProcessError as exception:
        logger.debug(exception.stderr.decode().strip(), __name__.upper())
        return False

def compress_image(output_path, output_image_quality=80):
    output_image_compression = round(31 - (output_image_quality * 0.31))
    commands = [ '-hwaccel', 'auto', '-i', output_path, '-q:v', str(output_image_compression), '-y', output_path ]
    return run_ffmpeg(commands)
