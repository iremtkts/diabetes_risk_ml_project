import logging
import sys

LOG_FORMAT = (
    "%(asctime)s | "
    "%(name)s | "
    "%(levelname)s | "
    "%(message)s"
)


def get_logger(name: str) -> logging.Logger:
    """
    Create and configure a logger instance.

    Args:
        name (str): Logger name.

    Returns:
        logging.Logger: Configured logger object.
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(LOG_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    logger.propagate = False

    return logger