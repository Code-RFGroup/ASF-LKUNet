import logging


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    # if not logger.handlers:
    formatter = logging.Formatter(
        "[%(levelname)s] %(message)s"
    )

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)


    sh = logging.StreamHandler()
    logger.propagate =False
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger