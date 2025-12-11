import logging
import random
import time

from openai import APIError, RateLimitError

from .exceptions import MaxRetriesExceededError

logger = logging.getLogger(__name__)

ERROR_MAX_RETRY = "max retries exceeded"


def retry_loop(func, max_retries=5, base_delay=1):
    logger.debug(
        "Starting retry loop with max_retries=%d, base_delay=%d",
        max_retries,
        base_delay,
    )
    last_exception: BaseException | None = None

    for attempt in range(max_retries):
        try:
            result = func()
        except (RateLimitError, APIError) as e:  # noqa: PERF203
            last_exception = e
            logger.warning("Attempt %d failed: %s", attempt + 1, str(e))
            if attempt == max_retries - 1:
                break

            sleep_time = base_delay * (2**attempt) + random.uniform(0, 0.5)  # noqa: S311
            logger.debug("Sleeping for %.2f seconds before retry", sleep_time)
            time.sleep(sleep_time)
        else:
            if attempt > 0:
                logger.info("Operation succeeded on attempt %d", attempt + 1)
            return result

    # If we reach here, all retries failed
    logger.error("All %d retry attempts failed", max_retries)
    raise last_exception or MaxRetriesExceededError(ERROR_MAX_RETRY)
