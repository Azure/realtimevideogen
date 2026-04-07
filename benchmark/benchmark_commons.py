import sys
import logging

from typing import Dict
from typing import Any

HEADERS_JSON = {
    "Content-Type": "application/json"
}


def setup_logging() -> None:
    """
    Set up logging configuration to log to stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def log_and_print(file_name: str, message: str) -> None:
    """
    Log a message to both stdout and a specified file.
    """
    print(message)
    with open(file_name, "a") as file_csv:
        file_csv.write(message + "\n")


class ServiceRequestInfo:
    def __init__(self, data_json: Dict[Any, Any]) -> None:
        pass

    def to_csv_str(self) -> str:
        return ""

    @staticmethod
    def get_csv_header() -> str:
        return ""
