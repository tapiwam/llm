
import os
import logging
from datetime import datetime
from pathlib import Path, PurePath

"""
Read in OpenAPI key from file and add to ENV
"""
def set_openai_key(key_file='./openai_key.txt'):
    api_key = open(key_file).read()
    os.environ['OPENAI_API_KEY'] = api_key
    logging.info('Set Open AI Key')

"""
Read in Huggingface key from file and add to ENV
"""
def set_huggingface_key(key_file='./hf_key.txt'):
    api_key = open(key_file).read()
    os.environ['HUGGINGFACE_API_KEY'] = api_key
    logging.info('Set Huggingface AI Key')

"""
Setup logging
"""
def setup_logging(name="rootLogger"):
    dt_format = "%Y%m%d%H%M%S"
    datestr = datetime.now().strftime(dt_format)
    logPath = './logs'
    logFileName = f'{logPath}/app_log_{datestr}.log'
    
    logFile = Path(logFileName)
    logFile.parent.mkdir(parents=True, exist_ok=True)
    logFile.touch()
    
    # logging.basicConfig(filename=, encoding='utf-8', level=logging.DEBUG)
    
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger(name)
    
    fileHandler = logging.FileHandler(logFileName)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.INFO)
    rootLogger.info('---Setup logging---')

    return rootLogger

