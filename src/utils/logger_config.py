import logging
from pathlib import Path

import toml
from opencensus.ext.azure.log_exporter import AzureLogHandler

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def _configure_azure_handler(logger):
    secrets_file = Path(__file__).parent.parent.joinpath("config.toml")
    try:
        secrets = toml.load(secrets_file)
        connection_string = secrets.get("azure", {}).get(
            "application_insights_connection_str"
        )
        if connection_string:
            azure_handler = AzureLogHandler(connection_string=connection_string)
            logger.addHandler(azure_handler)
            return azure_handler
    except FileNotFoundError:
        pass


def configure_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(LOG_FORMAT)
        handler.setFormatter(formatter)

        logger.addHandler(handler)

        _configure_azure_handler(logger)

    return logger
