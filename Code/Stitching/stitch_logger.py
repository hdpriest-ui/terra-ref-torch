import logging
from constant import LOG_LEVEL

def get_logger(name):
    """
    Create and configure a logger.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        # Set log level from constants
        logger.setLevel(LOG_LEVEL)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LOG_LEVEL)

        # Create formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(console_handler)

    return logger