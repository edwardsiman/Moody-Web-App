import os
import logging
import logging.config
import yaml
import time
from logging import Formatter

LOG_CONFIG = os.path.join(os.getcwd(), "logging.yml")

def setup_logging():
    with open(LOG_CONFIG, 'r') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

class GMTFormatter(Formatter):
    """Formatter that converts time to GMT
    """
    converter = time.gmtime