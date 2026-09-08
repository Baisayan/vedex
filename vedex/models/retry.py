import logging
import os

from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)


def retry(*, logger: logging.Logger, abort_exceptions: list[type[Exception]]) -> Retrying:
    return Retrying(
        reraise=True,
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(tuple(abort_exceptions)),
    )
