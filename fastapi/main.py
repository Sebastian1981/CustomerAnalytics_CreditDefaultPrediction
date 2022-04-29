from fastapi import FastAPI
import logging
from logging.config import dictConfig
from logger_config import log_config


app = FastAPI()
dictConfig(log_config)
logger = logging.getLogger("mlops")

@app.get('/health')
def health():
    logger.info("Health request received.")
    return "Service is online."