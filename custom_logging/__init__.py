import os
import logging
import logging.handlers
import time
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


class UTCTimeFormatter(logging.Formatter):
    converter = time.gmtime

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            s = time.asctime(ct)
        return s


def setup_logger(
    log_level,
    log_filename=None,
    max_log_files=100,
    max_log_size=None,
    utc_time=False,
    logger_name=__name__,
):
    """
    Setup a logger with a specific log level. The logger will write logs to a file, output logs to the console. All handlers use the same formatter, which includes the timestamp,
    logger name, log level, and log message in each log entry.

    Parameters:
    log_level (str): Level of logging. Could be 'debug', 'info', 'warning', 'error' or 'critical'.
    log_filename (str): Name of the log file.
    max_log_files (int): Maximum number of log files to keep.
    max_log_size (int): Maximum size of a log file in kilobytes. If the size reaches the limit, a new log file is created.

    Returns:
    logger (logging.Logger): Configured logger.

    Example Use:
    # Setup the logger
    logger = setup_logger(
        logger_name = "views",
        log_level="info",
        log_filename="pyagent.log",
        max_log_files=30,
        max_log_size=1024,
        utc_time=True,
    )

    # Log some messages
    logger.info("This is an info message")
    logger.debug("This is a debug message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    """

    # Create a custom logger
    logger = logging.getLogger(logger_name)

    # Map log_level from string to corresponding logging level
    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    # Set level of logging
    logger.setLevel(levels.get(log_level.lower(), logging.INFO))

    # Ensure the 'logs' folder exists
    # log_directory = './logs'
    # if not os.path.exists(log_directory):
    #     os.makedirs(log_directory)

    # Create formatters
    if utc_time:
        formatter = UTCTimeFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Create a file handler (skip if log_filename is None)
    if log_filename:
        if max_log_size:
            handler = RotatingFileHandler(
                log_filename, maxBytes=max_log_size * 1024, backupCount=max_log_files
            )
        else:
            handler = TimedRotatingFileHandler(
                log_filename, when="midnight", backupCount=max_log_files, utc=utc_time
            )
        handler.setLevel(logger.level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Create a console handler for logging to stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logger.level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
