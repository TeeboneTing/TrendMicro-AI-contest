
import os
import logging


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter('%(asctime)-24s %(name)-35s line:%(lineno)-5s %(message)s')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if os.path.exists('/log'):
        file_handle = logging.FileHandler('/log/log.txt')
        file_handle.setLevel(logging.INFO)
        file_handle.setFormatter(fmt)
        logger.addHandler(file_handle)

    return logger
