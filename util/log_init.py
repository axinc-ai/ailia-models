import sys
import datetime

from logging import getLogger, StreamHandler, FileHandler, Formatter
from logging import INFO, DEBUG


# ===== User Configuration ====================================================

# Log file name (if disable_file_handler is set to False)
now = datetime.datetime.now()
save_filename = now.strftime('%Y%m%d') + '.log'

# level: CRITICAL > ERROR > WARNING > INFO > DEBUG
log_level = INFO

# params
disable_stream_handler = False
disable_file_handler = True  # set False if you want to save text log file
display_date = False

# =============================================================================

# default logging format
if display_date:
    datefmt = '%Y/%m/%d %H:%M:%S'
    default_fmt = Formatter(
        '[%(asctime)s.%(msecs)03d] %(levelname)5s '
        '(%(process)d) %(filename)s: %(message)s',
        datefmt=datefmt
    )
else:
    default_fmt = Formatter(
        '%(levelname)5s %(filename)s (%(lineno)d) : %(message)s'
    )


logger = getLogger()

# remove duplicate handlers
if (logger.hasHandlers()):
    logger.handlers.clear()

# the level of logging passed to the Handler
logger.setLevel(log_level)

# set up stream handler
if not disable_stream_handler:
    try:
        # Rainbow Logging
        from rainbow_logging_handler import RainbowLoggingHandler  # noqa: E402
        color_msecs = ('green', None, True)
        stream_handler = RainbowLoggingHandler(
            sys.stdout, color_msecs=color_msecs, datefmt=datefmt
        )
        # msecs color
        stream_handler._column_color['.'] = color_msecs
        stream_handler._column_color['%(asctime)s'] = color_msecs
        stream_handler._column_color['%(msecs)03d'] = color_msecs
    except Exception:
        stream_handler = StreamHandler()

    # the level of output logging
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(default_fmt)
    logger.addHandler(stream_handler)

if not disable_file_handler:
    file_handler = FileHandler(filename=save_filename)

    # the level of output logging
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(default_fmt)
    logger.addHandler(file_handler)
