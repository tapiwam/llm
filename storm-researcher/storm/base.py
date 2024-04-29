from logging import config
import os, logging
from datetime import datetime
import pprint
from dotenv import load_dotenv, find_dotenv

def get_logger(name: str, console_level=logging.INFO, file_level=logging.DEBUG, log_file:str|None = None) -> logging.Logger:
        
    # Log file with current date
    log_file = f'./logs/storm_log_{datetime.now().strftime("%Y%m%d")}.log' if log_file is None else log_file

    # Create parent directory for log file and create an empty file if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        open(log_file, 'a').close()

    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    logger = logging.getLogger(name)

    # Console logger
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    # Colsole level to WARN ONLY
    consoleHandler.setLevel(logging.INFO)


    # Add file logger
    fileHandler = logging.FileHandler(log_file)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    # Set level
    logger.setLevel(logging.INFO)
    
    return logger
