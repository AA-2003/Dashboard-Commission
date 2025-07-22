import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger(name="dashboard"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # ساخت مسیر logs اگر وجود نداشت
    os.makedirs("logs", exist_ok=True)
    handler = RotatingFileHandler("logs/dashboard.log", maxBytes=1000000, backupCount=3)
    
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger
