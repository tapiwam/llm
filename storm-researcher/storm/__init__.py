import os, logging
from datetime import datetime
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file by default
load_dotenv(find_dotenv())


from .models import *
from .fns import *
from .llm_tools import *

from . import prompts

# ==========================================
# Global tools and variables
# ==========================================

MAX_INTERVIEW_QUESTIONS = 7
TAGS_TO_EXTRACT = [ "p", "h1", "h2", "h3"]

wikipedia_retriever = get_wikipedia_retriever()


# ==========================================
# Setup logging
# ==========================================

# Log file with current date
log_file = f'./logs/storm_log_{datetime.now().strftime("%Y%m%d")}.log'

# Create parent directory for log file and create an empty file if it doesn't exist
log_dir = os.path.dirname(log_file)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    open(log_file, 'a').close()

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger(__name__)

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
