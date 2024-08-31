from logging import INFO, Logger, getLogger, basicConfig


def create_logger(name: str, level: str | int = INFO) -> Logger:
    """Get a logger given `name`, and set the `level`"""
    basicConfig()
    logger = getLogger(name)
    logger.setLevel(level)
    return logger
